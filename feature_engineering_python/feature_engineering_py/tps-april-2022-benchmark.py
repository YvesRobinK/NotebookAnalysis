#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Tabular Playground Series April 2022! #

# This notebook outlines a way of applying traditional machine learning algorithms to time series classification problems by generating a set of features for each series with the `tsfresh` library, adapted from [this tutorial](https://tsfresh.readthedocs.io/en/latest/text/sklearn_transformers.html). There are many algorithms specific to this task, however, some of which you can read about on this [Time Series Classification](https://www.timeseriesclassification.com/algorithm.php) website.

# In[1]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython import get_ipython

warnings.filterwarnings('ignore')

plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    titlesize=18,
    titleweight="bold",
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
get_ipython().config.InlineBackend.figure_format = 'retina'


# # Data #

# In[2]:


data_dir = Path('../input/tabular-playground-series-apr-2022')
df_train = pd.read_csv(data_dir / 'train.csv', index_col=['sequence', 'subject', 'step'])
labels_train = pd.read_csv(data_dir / 'train_labels.csv', index_col='sequence').squeeze()

display(df_train)
display(labels_train)


# Let's take a look at sequence of sensor data.

# In[3]:


SEQ = 0
df_train.loc[SEQ].plot(subplots=True, sharex=True, figsize=(18, 1.5*13));


# # Train #

# In[4]:


def score(model, X_test, y_test, X_train=None, y_train=None, fitted=True):
    from sklearn import metrics

    if not fitted:
        model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]

    print('Acc\t', metrics.accuracy_score(y_test, y_pred.round()))
    print('AUC:\t', metrics.roc_auc_score(y_test, y_pred))
    print('AP:\t', metrics.average_precision_score(y_test, y_pred))
    print('Rec:\t', metrics.recall_score(y_test, y_pred.round()))
    print('Prec:\t', metrics.precision_score(y_test, y_pred.round()))
    print('F1:\t', metrics.f1_score(y_test, y_pred.round()))


# In[5]:


from sklearn.model_selection import GroupShuffleSplit

N_SEQS = 12000  # Only use a subset of the sequences for the sake of time
TEST_SIZE = 0.2

sequences, subjects = (
    df_train
    .reset_index()
    .loc[:, ['sequence', 'subject']]
    .drop_duplicates()
    .to_numpy()
    [:N_SEQS]
    .T
)

splitter = GroupShuffleSplit(test_size=TEST_SIZE, n_splits=1, random_state = 0)
seq_train, seq_valid = next(splitter.split(sequences, groups=subjects))

X_train, X_valid = df_train.loc[seq_train], df_train.loc[seq_valid]
y_train, y_valid = labels_train.loc[seq_train], labels_train.loc[seq_valid]

display(X_train)
display(y_train)


# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tsfresh.transformers import RelevantFeatureAugmenter

# Format time series for tsfresh pipeline 
N_SENSORS = 2  # And only use a couple of the sensor readings
features = [f'sensor_{k:02d}' for k in range(N_SENSORS)]
X_train = X_train.reset_index().loc[:, ['sequence', 'step'] + features]
X_valid = X_valid.reset_index().loc[:, ['sequence', 'step'] + features]

# Extra (non-time series) features go here.
# There are none, so we'll create dummy frames to satisfy the arguments of the fit/predict methods.
Xtra_train = pd.DataFrame(np.zeros_like(y_train), index=y_train.index)
Xtra_valid = pd.DataFrame(np.zeros_like(y_valid), index=y_valid.index)

model = Pipeline([
    ('augmenter', RelevantFeatureAugmenter(column_id='sequence', column_sort='step')),
    ('rf', RandomForestClassifier(n_jobs=-1)),
])
model.set_params(augmenter__timeseries_container=X_train)
model.fit(Xtra_train, y_train)

model.set_params(augmenter__timeseries_container=X_valid)
score(model, Xtra_valid, y_valid)


# # Infer #

# In[7]:


df_test = pd.read_csv(data_dir / 'test.csv', index_col=['sequence', 'subject', 'step'])
sample_submission = pd.read_csv(data_dir / 'sample_submission.csv', index_col=['sequence'])

X_test = df_test.reset_index().loc[:, ['sequence', 'step'] + features]
Xtra_test = pd.DataFrame(np.zeros_like(sample_submission), index=sample_submission.index)

model.set_params(augmenter__timeseries_container=X_test)
sample_submission['state'] = model.predict_proba(Xtra_test)[:, 1]

sample_submission.to_csv('submission.csv')

