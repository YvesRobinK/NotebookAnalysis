#!/usr/bin/env python
# coding: utf-8

# # How to Write Powerful Code Others Envy With Custom Sklearn Transformers
# ## Do everything in Sklearn and everyone will be happy
# ![](https://cdn-images-1.medium.com/max/1440/1*y3zaj2WueHb-j6hPwugEGA.jpeg)

# # Setup

# In[1]:


import logging
import time
import warnings

import catboost as cb
import joblib
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# # Introduction

# Single `fit`, single `predict` - how awesome would that be?
# 
# You get the data, fit your pipeline just one time, and it takes care of everything - preprocessing, feature engineering, modeling, everything. All you have to do is call predict and have the output.
# 
# What kind of pipeline is *that* powerful? Yes, Sklearn has many transformers, but it doesn't have one for every imaginable preprocessing scenario. So, is such a pipeline a *pipe* dream?
# 
# Absolutely not. Today, we will learn how to create custom Sklearn transformers that enable you to integrate virtually any function or data transformation into Sklearn's Pipeline classes.

# # What are Sklearn pipelines?

# Below is a simple pipeline that imputes the missing values in numeric data, scales them, and fits an XGBRegressor to `X`, `y`:

# ```python
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# import xgboost as xgb
# 
# xgb_pipe = make_pipeline(
#                 SimpleImputer(strategy='mean'),
#                 StandardScaler(),
#                 xgb.XGBRegressor()
#             )
# 
# _ = xgb_pipe.fit(X, y)
# ```

# I have talked at length about the nitty-gritty of Sklearn pipelines and their benefits in an [older post](https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d). The most notable advantages are their ability to collapse all preprocessing and modeling steps into a single estimator, preventing data leakage by never calling `fit` on validation sets and an added bonus that makes the code concise, reproducible, and modular.
# 
# But this whole idea of atomic, neat pipelines breaks when we need to perform operations that are not built into Sklearn as estimators. For example, what if you need to extract regex patterns to clean text data? What do you do if you want to create a new feature combining existing ones based on domain knowledge?
# 
# To preserve all the benefits that come with pipelines, you need a way to integrate your custom preprocessing and feature engineering logic into Sklearn. That's where custom transformers come into play.

# # Integrating simple functions with `FunctionTransformer`

# In this month's (September) TPS Competition on Kaggle, one of the ideas that boosted model performance significantly was adding the number of missing values in a row as a new feature. This is a custom operation, not implemented in Sklearn, so let's create a function to achieve that after importing the data:

# In[2]:


tps_df = pd.read_csv("../input/tabular-playground-series-sep-2021/train.csv")
tps_df.head()


# In[3]:


tps_df.shape


# In[4]:


# Find the number of missing values across rows
tps_df.isnull().sum(axis=1)


# Let's create a function that takes a DataFrame as input and implements the above operation:

# In[5]:


def num_missing_row(X: pd.DataFrame, y=None):
    # Calculate some metrics across rows
    num_missing = X.isnull().sum(axis=1)
    num_missing_std = X.isnull().std(axis=1)

    # Add the above series as a new feature to the df
    X["#missing"] = num_missing
    X["num_missing_std"] = num_missing_std

    return X


# Now, adding this function into a pipeline is just as easy as passing it to the `FunctionTransformer`:

# In[6]:


from sklearn.preprocessing import FunctionTransformer

num_missing_estimator = FunctionTransformer(num_missing_row)


# Passing a custom function to `FunctionTransformer` creates an estimator with `fit`, `transform` and `fit_transform` methods:

# In[7]:


# Check number of columns before
print(f"Number of features before preprocessing: {len(tps_df.columns)}")

# Apply the custom estimator
tps_df = num_missing_estimator.transform(tps_df)
print(f"Number of features after preprocessing: {len(tps_df.columns)}")


# ince we have a simple function, no need to call `fit` as it just returns the estimator untouched. The only requirement of `FunctionTransformer` is that the passed function should accept the data as its first argument. Optionally, you can pass the target array as well if you need it inside the function:

# ```python
# # FunctionTransformer signature
# def custom_function(X, y=None):
#     ...
# 
# estimator = FunctionTransformer(custom_function)  # no errors
# 
# custom_pipeline = make_pipeline(StandardScaler(), estimator, xgb.XGBRegressor())
# custom_pipeline.fit(X, y)
# ```

# `FunctionTransformer` also accepts an inverse of the passed function if you ever need to revert the changes:

# In[8]:


def custom_function(X, y=None):
    ...


def inverse_of_custom(X, y=None):
    ...


estimator = FunctionTransformer(func=custom_function, inverse_func=inverse_of_custom)


# Check out the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) for details on other arguments.

# # Integrating more complex preprocessing steps with custom transformers

# > This section assumes some knowledge of Python object-oriented programming (OOP). Specifically, the basics of creating classes and inheritance. If you are not already down with those, check out my [Python OOP series](https://ibexorigin.medium.com/list/objectoriented-programming-essentials-for-data-scientists-cf2ff3dc9fc9?source=user_lists---------1-------cf2ff3dc9fc9---------------------), written for data scientists.

# One of the most common scaling options for skewed data is a logarithmic transform. But here is a caveat: if a feature has even a single 0, the transformation with `np.log` or Sklearn's `PowerTransformer` return an error.
# 
# So, as a workaround, people add 1 to all samples and then apply the transformation. If the transformation is performed on the target array, you will also need an inverse transform. For that, after making predictions, you need to use the exponential function and subtract 1. Here is what it looks like in code:

# ```python
# y_transformed = np.log(y + 1)
# 
# _ = model.fit(X, y_transformed)
# preds = np.exp(model.predict(X, y_transformed) - 1)
# ```

# This works, but we have the same old problem - we can't include this into a pipeline out of the box. Sure, we could use our newfound friend `FunctionTransformer`, but it is not well-suited for more complex preprocessing steps such as this.
# 
# Instead, we will write a custom transformer class and create the `fit`, `transform` functions manually. In the end, we will again have a Sklearn-compatible estimator that we can pass into a pipeline. Let's start:

# In[9]:


from sklearn.base import BaseEstimator, TransformerMixin


class CustomLogTransformer(BaseEstimator, TransformerMixin):
    pass


# We first create a class that inherits from `BaseEstimator` and `TransformerMixin` classes of `sklearn.base`. Inheriting from these classes allows Sklearn pipelines to recognize our classes as custom estimators. 
# 
# Then, we will write the `__init__` method, where we just initialize an instance of `PowerTransformer`:

# In[10]:


from sklearn.preprocessing import PowerTransformer


class CustomLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = PowerTransformer()


# Next, we write the `fit` where we add 1 to all features in the data and fit the PowerTransformer:

# In[11]:


class CustomLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = PowerTransformer()

    def fit(self, X, y=None):
        X_copy = np.copy(X) + 1
        self._estimator.fit(X_copy)

        return self


# The `fit` method should return the transformer itself, which is done by returning `self`. Let's test what we have done so far:

# In[12]:


custom_log = CustomLogTransformer()
custom_log.fit(tps_df)


# Next, we have the `transform`, in which we just use the `transform` method of PowerTransformer after adding 1 to the passed data:

# In[13]:


class CustomLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = PowerTransformer()

    def fit(self, X, y=None):
        X_copy = np.copy(X) + 1
        self._estimator.fit(X_copy)

        return self

    def transform(self, X):
        X_copy = np.copy(X) + 1

        return self._estimator.transform(X_copy)


# Let's make another check:

# In[14]:


custom_log = CustomLogTransformer()
custom_log.fit(tps_df)

transformed_tps = custom_log.transform(tps_df)


# In[15]:


transformed_tps[:5, :5]


# Working as expected. Now, as I said earlier, we need a method for reverting the transform:

# In[16]:


class CustomLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._estimator = PowerTransformer()

    def fit(self, X, y=None):
        X_copy = np.copy(X) + 1
        self._estimator.fit(X_copy)

        return self

    def transform(self, X):
        X_copy = np.copy(X) + 1

        return self._estimator.transform(X_copy)

    def inverse_transform(self, X):
        X_reversed = self._estimator.inverse_transform(np.copy(X))

        return X_reversed - 1


# We also could have used `np.exp` instead of `inverse_transform`. Now, let's make a final check:

# In[17]:


custom_log = CustomLogTransformer()

tps_transformed = custom_log.fit_transform(tps_df)
tps_inversed = custom_log.inverse_transform(tps_transformed)


# > But wait! We didn't write `fit_transform` - where did that come from? It is simple - when you inherit from `BaseEstimator` and `TransformerMixin`, you get a `fit_transform` method for free. 
# 
# After the inverse transform, you can compare it with the original data:

# In[18]:


tps_df.values[:5, 5]


# In[19]:


tps_inversed[:5, 5]


# Now, we have a custom transformer ready to be included in a pipeline. Let's put everything together:

# In[20]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

xgb_pipe = make_pipeline(
    FunctionTransformer(num_missing_row),
    SimpleImputer(strategy="constant", fill_value=-99999),
    CustomLogTransformer(),
    xgb.XGBClassifier(
        n_estimators=1000, tree_method="gpu_hist", objective="binary:logistic"
    ),
)

X, y = tps_df.drop("claim", axis=1), tps_df[["claim"]].values.flatten()
split = train_test_split(X, y, test_size=0.33, random_state=1121218)
X_train, X_test, y_train, y_test = split


# In[21]:


xgb_pipe.fit(X_train, y_train)
preds = xgb_pipe.predict_proba(X_test)

roc_auc_score(y_test, preds[:, 1])


# Even though log transform actually hurt the score, we got our custom pipeline working!
# 
# In short, the signature of your custom transformer class should be like this:

# In[22]:


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def inverse_transform(self):
        pass


# This way, you get `fit_transform` for free. If you don't need any of `__init__`, `fit`, `transform` or `inverse_transform` methods, omit them and the parent Sklearn classes take care of everything. The logic of these methods are entirely up to your coding skills and needs.

# # Wrapping up...

# Writing good code is a skill developed over time. You will realize that a big part of it comes from using the existing tools and libraries at the right time and place, without having to reinvent the wheel.
# 
# One of such tools is Sklearn pipelines and custom transformers are just extensions of them. Use them well and you will produce quality code with little effort.

# # You might also be interested...
# 
# https://towardsdatascience.com/how-to-work-with-million-row-datasets-like-a-pro-76fb5c381cdd
# 
# https://towardsdatascience.com/how-to-beat-the-heck-out-of-xgboost-with-lightgbm-comprehensive-tutorial-5eba52195997
# 
# https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
# 
# https://towardsdatascience.com/tired-of-clich%C3%A9-datasets-here-are-18-awesome-alternatives-from-all-domains-196913161ec9
# 
# https://towardsdatascience.com/love-3blue1brown-animations-learn-how-to-create-your-own-in-python-in-10-minutes-8e0430cf3a6d
