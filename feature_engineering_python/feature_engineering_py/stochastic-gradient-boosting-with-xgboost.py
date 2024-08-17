#!/usr/bin/env python
# coding: utf-8

# # If you like this kernel Greatly Appreciate to UPVOTE .
# 
# 
# 
# # Stochastic Gradient Boosting with XGBoost 
# 
# A simple technique for ensembling decision trees involves training trees on subsamples of the training dataset.
# 
# Subsets of the the rows in the training data can be taken to train individual trees called bagging. When subsets of rows of the training data are also taken when calculating each split point, this is called random forest.
# 
# These techniques can also be used in the gradient tree boosting model in a technique called stochastic gradient boosting.
# 
# In this kernel I will be demonstrating stochastic gradient boosting and how to tune the sampling parameters using XGBoost with scikit-learn in Python.
# 
# After reading this kernel we will get to know the following point in detail:
# 
# - The rationale behind training trees on subsamples of data and how this can be used in gradient boosting.
# - How to tune row-based subsampling in XGBoost using scikit-learn?
# - How to tune column-based subsampling by both tree and split-point in XGBoost?
# 
# ## What is Stochastic Gradient Boosting? Let's understand the concept in detail
# 
# Gradient boosting is a greedy procedure.
# 
# New decision trees are added to the model to correct the residual error of the existing model.
# 
# Each decision tree is created using a greedy search procedure to select split points that best minimize an objective function. This can result in trees that use the same attributes and even the same split points again and again.
# 
# Bagging is a technique where a collection of decision trees are created, each from a different random subset of rows from the training data. The effect is that better performance is achieved from the ensemble of trees because the randomness in the sample allows slightly different trees to be created, adding variance to the ensembled predictions.
# 
# Random forest takes this one step further, by allowing the features (columns) to be subsampled when choosing split points, adding further variance to the ensemble of trees.
# 
# These same techniques can be used in the construction of decision trees in gradient boosting in a variation called stochastic gradient boosting.
# 
# It is common to use aggressive sub-samples of the training data such as 40% to 80%.
# 
# # Overview
# 
# In this kernel we are going to look at the effect of different subsampling techniques in gradient boosting.
# 
# We will tune three different flavors of stochastic gradient boosting supported by the XGBoost library in Python, specifically:
# 
# **1.  Subsampling of rows in the dataset when creating each tree.**
# 
# **2. Subsampling of columns in the dataset when creating each tree.**
# 
# **3. Subsampling of columns for each split in the dataset when creating each tree.**
# 
# ## Dataset
# 
# We will use the **Otto Group Product Classification Challenge** dataset available in Kaggle which is available for free.
# This dataset describes the 93 obfuscated details of more than 61,000 products grouped into 10 product categories (e.g. fashion, electronics, etc.). Input attributes are counts of different events of some kind.The Otto Group is one of the world’s biggest e-commerce companies, with subsidiaries in more than 20 countries, including Crate & Barrel (USA), Otto.de (Germany) and 3 Suisses (France) selling millions of products worldwide every day, with several thousand products being added to our product line.
# 
# #### Data fields
# - id - an anonymous id unique to a product
# - feat_1, feat_2, ..., feat_93 - the various features of a product
# - target - the class of a product
# 
# ## Problem Description
# 
# The goal is to make predictions for new products as an array of probabilities for each of the 10 categories and models are evaluated using multiclass logarithmic loss (also called cross entropy).
# 
# ## Solution Approach
# 
# As mentioned above let us look at the appraoches one by one.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


# Load data

# In[ ]:


data = read_csv('../input/train.csv')
dataset = data.values


# Split the data into X and y

# In[ ]:


X = dataset[:,0:94]
y = dataset[:,94]


# Encode string class values as integers

# In[ ]:


label_encoded_y = LabelEncoder().fit_transform(y)
model = XGBClassifier()


# ### 1.  Subsampling of rows in the dataset when creating each tree.
# 
# Row subsampling involves selecting a random sample of the training dataset without replacement.
# 
# Row subsampling can be specified in the scikit-learn wrapper of the XGBoost class in the subsample parameter. The default is 1.0 which is no sub-sampling.
# 
# We can use the grid search capability built into scikit-learn to evaluate the effect of different subsample values from 0.1 to 1.0 on the Otto dataset.
# 
# There are 9 variations of subsample and each model will be evaluated using 10-fold cross validation, meaning that 9×10 or 90 models need to be trained and tested.
# 
# Perform K Fold cross validation using the GridSearchCV 

# In[ ]:


subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(subsample=subsample)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)


# Summarize results

# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# We can see that the best results achieved were 0.3, or training trees using a 30% sample of the training dataset.
# 
# We can plot these mean and standard deviation log loss values to get a better understanding of how performance varies with the subsample value.

# In[ ]:


pyplot.errorbar(subsample, means, yerr=stds)
pyplot.title("XGBoost subsample vs Log Loss")
pyplot.xlabel('subsample')
pyplot.ylabel('Log Loss')


# We can see that indeed 30% has the best mean performance, but we can also see that as the ratio increased, the variance in performance grows quite markedly.
# 
# It is interesting to note that the mean performance of all subsample values outperforms the mean performance without subsampling (**subsample**=1.0).
# 
# **2. Subsampling of columns in the dataset when creating each tree.**
# 
# We can also create a random sample of the features (or columns) to use prior to creating each decision tree in the boosted model.
# 
# In the XGBoost wrapper for scikit-learn, this is controlled by the **colsample_bytree** parameter.
# 
# The default value is 1.0 meaning that all columns are used in each decision tree. We can evaluate values for **colsample_bytree** between 0.1 and 1.0 incrementing by 0.1.
# 
# Perform K Fold cross validation using the GridSearchCV

# In[ ]:


colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(colsample_bytree=colsample_bytree)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)


# Summarize results

# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# We can see that the best performance for the model was **colsample_bytree**=1.0. This suggests that subsampling columns on this problem does not add value.

# In[ ]:


pyplot.errorbar(colsample_bytree, means, yerr=stds)
pyplot.title("XGBoost colsample_bytree vs Log Loss")
pyplot.xlabel('colsample_bytree')
pyplot.ylabel('Log Loss')


# Plotting the results, we can see the performance of the model plateau (at least at this scale) with values between 0.5 to 1.0.
# 
# 
# 
# **3. Subsampling of columns for each split in the dataset when creating each tree.**
# 
# 
# Rather than subsample the columns once for each tree, we can subsample them at each split in the decision tree. In principle, this is the approach used in random forest.
# 
# We can set the size of the sample of columns used at each split in the colsample_bylevel parameter in the XGBoost wrapper classes for scikit-learn.
# 
# As before, we will vary the ratio from 10% to the default of 100%.

# In[ ]:


# grid search
model = XGBClassifier()
colsample_bylevel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
param_grid = dict(colsample_bylevel=colsample_bylevel)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# We can see that the best results were achieved by setting colsample_bylevel to 70%, resulting in an (inverted) log loss of -0.001062, which is better than -0.001239 seen when setting the per-tree column sampling to 100%.
# 
# This suggest to not give up on column subsampling if per-tree results suggest using 100% of columns, and to instead try per-split column subsampling.
# 
# We can plot the performance of each colsample_bylevel variation. The results show relatively low variance and seemingly a plateau in performance after a value of 0.3 at this scale.

# In[ ]:


pyplot.errorbar(colsample_bylevel, means, yerr=stds)
pyplot.title("XGBoost colsample_bylevel vs Log Loss")
pyplot.xlabel('colsample_bylevel')
pyplot.ylabel('Log Loss')


# # Summary
# 
# In this kernel we have discovered what is stochastic gradient boosting with XGBoost in Python.
# 
# Specifically, we learned:
# 
# - About stochastic boosting and how you can subsample your training data to improve the generalization of your model
# - How to tune row subsampling with XGBoost in Python and scikit-learn.
# - How to tune column subsampling with XGBoost both per-tree and per-split.
# 
# # If you like this kernel Greatly Appreciate to UPVOTE .

# In[ ]:




