#!/usr/bin/env python
# coding: utf-8

# ### Hi, in this kernel we'll make a baseline submission using sklearn pipelines instead of functions. Some of the advantages:
# * Less and more readable code
# * Faster feature engineering
# * More automation

# ## Download data

# In[1]:


import pandas as pd
import numpy as np
import os


# **If you're working on a local machine, it's always a good practice to write a function for data downloading. It can be as easy as the one shown below.**

# In[2]:


# ALICE_PATH = os.path.join("datasets", "alice")

# def load_alice_data(alice_path=ALICE_PATH):
#     csv_path_train = os.path.join(alice_path, "train_sessions.csv")
#     csv_path_test = os.path.join(alice_path, "test_sessions.csv")
#     return pd.read_csv(csv_path_train, index_col='session_id', parse_dates=['time1']), \
#             pd.read_csv(csv_path_test, index_col='session_id', parse_dates=['time1'])

# df_train, df_test = load_alice_data()


# **Here, in the kernel, we'll do this manually.**

# In[3]:


df_train = pd.read_csv("../input/train_sessions.csv", index_col='session_id', parse_dates=['time1'])
df_test = pd.read_csv("../input/test_sessions.csv", index_col='session_id', parse_dates=['time1'])


# ## Clean data

# In[4]:


df_train.head()


# **First of all we should notice, that our data is time dependent. Leaving it shuffled will cause some problems later on during the cross-validation, so let's sort it right away.**

# In[5]:


df_train = df_train.sort_values(by="time1")


# **Now let's take a look at the column data types.**

# In[6]:


df_train.info()


# **It seems like only the first time column is in datetime format. Better convert the rest now, otherwise you'll have a problem with the datetime arithmetic, for example, when calculation session duration as a feature.**

# In[7]:


for i in range(2, 11):
    df_train['time{}'.format(i)] = pd.to_datetime(df_train['time{}'.format(i)])
for i in range(2, 11):
    df_test['time{}'.format(i)] = pd.to_datetime(df_test['time{}'.format(i)])


# ## Pipeline configuration

# **I'll skip EDA and feature engineering for now – there are plenty of great kernels in this competition, that cover those in depth. Instead I'll try to show you another way of preparing data and training models – using the sklearn ``Pipeline()`` class, instead of good old functions. This approach is based more on object oriented, rather than procedural programming, which in fact reduces the length of your code quite a lot.**

# In[8]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


# **The key idea here is to write your own classes for data transformation. They must follow a pretty simple template: they should inherit from two base classes – ``BaseEstimator``, ``TransformerMixin`` – and have a ``fit()`` and ``transform()`` methods, which take the dataset (X) as input and have a y value set to None. In pipelines the ``__init__()`` method is not neccessary.**

# In[9]:


class DataPreparator(BaseEstimator, TransformerMixin):
    """
    Fill NaN with zero values.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        sites = ['site%s' % i for i in range(1, 11)]
        return X[sites].fillna(0).astype('int')


# In[10]:


class ListPreparator(BaseEstimator, TransformerMixin):
    """
    Prepare a CountVectorizer friendly 2D-list from data.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.values.tolist()
        # Convert dataframe rows to strings
        return [" ".join([str(site) for site in row]) for row in X]


# **One of the great advantages of pipelines, is that you can use classes for adding features too. You can test new features quite easilly by adding/removing some of them from the class, and then just running the whole pipeline. **

# In[11]:


class AttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add new attributes to training and test set.
    """
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        # intraday features
        hour = X['time1'].apply(lambda ts: ts.hour)
        morning = ((hour >= 7) & (hour <= 11)).astype('int')
        day = ((hour >= 12) & (hour <= 18)).astype('int')
        evening = ((hour >= 19) & (hour <= 23)).astype('int')
        
        # season features
        month = X['time1'].apply(lambda ts: ts.month)
        summer = ((month >= 6) & (month <= 8)).astype('int')
        
        # day of the week features
        weekday = X['time1'].apply(lambda ts: ts.weekday()).astype('int')
        
        # year features
        year = X['time1'].apply(lambda ts: ts.year).astype('int')
        
        X = np.c_[morning.values, day.values, evening.values, summer.values, weekday.values, year.values]
        return X


# In[12]:


class ScaledAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add new features, that should be scaled.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # session time features
        times = ['time%s' % i for i in range(1, 11)]
        # session duration: take to the power of 1/5 to normalize the distribution
        session_duration = (X[times].max(axis=1) - X[times].min(axis=1)).astype('timedelta64[ms]').astype(int) ** 0.2
        # number of sites visited in a session
        number_of_sites = X[times].isnull().sum(axis=1).apply(lambda x: 10 - x)
        # average time spent on one site during a session
        time_per_site = (session_duration / number_of_sites) ** 0.2
        
        X = np.c_[session_duration.values]
        return X


# **Once we've written the classes, we want to combine them into a pipeline. The ``Pipeline()`` class will call ``transform()`` methods on each one of them and return the transformed dataset, which you can pass to another pipeline as many times as you want. Here we have three separate pipelines with different purposes: ``vectorizer_pipeline`` prepares data for ``CountVectorizer()`` class, ``attributes_pipeline`` adds features and ``scaled_attributes_pipeline`` adds scaled features. **

# In[13]:


vectorizer_pipeline = Pipeline([
    ("preparator", DataPreparator()),
    ("list_preparator", ListPreparator()),
    ("vectorizer", CountVectorizer(ngram_range=(1, 3), max_features=50000))
])

attributes_pipeline = Pipeline([
    ("adder", AttributesAdder())
])

scaled_attributes_pipeline = Pipeline([
    ("adder", ScaledAttributesAdder()),
    ("scaler", StandardScaler())
])


# **Finally you can combine these pipelines using ``FeatureUnion()`` class, which will merge the resulting datasets from each pipeline. **

# In[14]:


full_pipeline = FeatureUnion(transformer_list=[
('vectorizer_pipeline', vectorizer_pipeline),
('attributes_pipeline', attributes_pipeline),
('scaled_attributes_pipeline', scaled_attributes_pipeline)
])


# **All you need to do at the end, is just call ``fit_transform()`` or ``transform()`` methods on the ``full_pipeline`` and pass them your original datasets.**

# In[15]:


X_train = full_pipeline.fit_transform(df_train)
X_test = full_pipeline.transform(df_test)

y_train = df_train["target"].astype('int').values


# ## Cross-validation and submitting results

# In[16]:


from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# **As @yorko introduced, we use time-aware cross-validation scheme.**

# In[17]:


time_split = TimeSeriesSplit(n_splits=10)

logit = LogisticRegression(C=1, random_state=42, solver='liblinear')

cv_scores = cross_val_score(logit, X_train, y_train, cv=time_split, 
                        scoring='roc_auc', n_jobs=1)

cv_scores.mean()


# **Finally, train your model on the whole train set and write a function for submitting results.**

# In[18]:


logit.fit(X_train, y_train)


# In[19]:


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[20]:


logit_test_pred = logit.predict_proba(X_test)[:, 1]

write_to_submission_file(logit_test_pred, 'submit.csv') # 0.95191


# **As you can see, by using the pipeline template you can write fewer lines of code and understand more, what's going on in your data preparation workflow. You can test features much faster. But the greatest thing about pipelines is that you can generalize the data preparation and training process by changing the dataset you use as a the pipeline input. For example, you can use the same pipeline for transforming your train and test set. And even more: you can automate the whole process, when new data becomes available.**
# 
# **Good luck with Alice!**
