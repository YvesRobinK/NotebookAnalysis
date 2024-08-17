#!/usr/bin/env python
# coding: utf-8

# # 📦 Import necessary libraries

# In[1]:


import numpy as np             # 🧮 NumPy for numerical operations
import pandas as pd            # 🐼 Pandas for data manipulation
from itertools import groupby  # 🔄 Import the 'groupby' function from itertools
import gc                      # 🗑️ Import the garbage collection module


# # 📊 Load the training data 

# In[2]:


# 📊 Load the training data from a Parquet file into a DataFrame
train = pd.read_parquet("/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train_multi.parquet")


# # 🛠️ Defining Features

# In[3]:


# 🛠️ Define a function to create additional features in the DataFrame
def make_features(df):
    # 🕒 Parse the timestamp and create an "hour" feature
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df["hour"] = df["timestamp"].dt.hour
    
    periods = 20
    
    # Feature engineering for "anglez"
    df["anglez"] = abs(df["anglez"])
    df["anglez_diff"] = df.groupby('series_id')['anglez'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["anglez_rolling_mean"] = df["anglez"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_rolling_max"] = df["anglez"].rolling(periods, center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_rolling_std"] = df["anglez"].rolling(periods, center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling_mean"] = df["anglez_diff"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["anglez_diff_rolling_max"] = df["anglez_diff"].rolling(periods, center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    
    # Feature engineering for "enmo"
    df["enmo_diff"] = df.groupby('series_id')['enmo'].diff(periods=periods).fillna(method="bfill").astype('float16')
    df["enmo_rolling_mean"] = df["enmo"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_max"] = df["enmo"].rolling(periods, center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_rolling_std"] = df["enmo"].rolling(periods, center=True).std().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling_mean"] = df["enmo_diff"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill").astype('float16')
    df["enmo_diff_rolling_max"] = df["enmo_diff"].rolling(periods, center=True).max().fillna(method="bfill").fillna(method="ffill").astype('float16')
    
    return df

# 📄 Define a list of feature names to be used later
features = ["hour", "anglez", "anglez_rolling_mean", "anglez_rolling_max", "anglez_rolling_std", "anglez_diff",
            "anglez_diff_rolling_mean", "anglez_diff_rolling_max", "enmo", "enmo_rolling_mean",
            "enmo_rolling_max", "enmo_rolling_std", "enmo_diff", "enmo_diff_rolling_mean", "enmo_diff_rolling_max"]


# # 🔄 Training

# In[4]:


# 🛠️ Apply the feature engineering function to the training data
train = make_features(train)

# 📊 Prepare the training data and target variable
X_train = train[features]  # Features
y_train = train["awake"]  # Target variable

# 🗑️ Delete the intermediate 'train' DataFrame to save memory and perform garbage collection
del train
gc.collect()


# # ⏱️ Model Training

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# 🧮 Import the GradientBoostingClassifier from sklearn\nfrom sklearn.ensemble import GradientBoostingClassifier\n\n# 🚀 Create a GradientBoostingClassifier with specified parameters\nclassifier = GradientBoostingClassifier(min_samples_leaf=100, n_estimators=50)\n\n# 🏋️\u200d♂️ Fit the classifier on the training data\nclassifier.fit(X_train, y_train)\n\n# 🗑️ Delete X_train and y_train to save memory and perform garbage collection\ndel X_train, y_train\ngc.collect()\n')


# # 🧠 Predictions

# In[6]:


# 📄 Load the test data from a Parquet file into a DataFrame
test = pd.read_parquet("/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet")

# 🛠️ Apply the feature engineering function to the test data
test = make_features(test)

# 📊 Prepare the test features
X_test = test[features]

# 🧠 Predict the probabilities for "not_awake" and "awake" using the trained classifier
test["not_awake"] = classifier.predict_proba(X_test)[:, 0]
test["awake"] = classifier.predict_proba(X_test)[:, 1]


# # 🔍Processing

# In[7]:


# 📊 Smooth the predictions for "score" and "smooth" using rolling mean
smoothing_length = 2 * 251
test["score"] = test["awake"].rolling(smoothing_length, center=True).mean().fillna(method="bfill").fillna(method="ffill")
test["smooth"] = test["not_awake"].rolling(smoothing_length, center=True).mean().fillna(method="bfill").fillna(method="ffill")

# 🔄 Re-binarize the "smooth" predictions
test["smooth"] = test["smooth"].round()

# 🔍 Define a function to extract events from the smoothed predictions
def get_event(df):
    lstCV = zip(df.series_id, df.smooth)
    lstPOI = []
    for (c, v), g in groupby(lstCV, lambda cv: (cv[0], cv[1] != 0 and not pd.isnull(cv[1]))):
        llg = sum(1 for item in g)
        if v is False:
            lstPOI.extend([0] * llg)
        else:
            lstPOI.extend(['onset'] + (llg - 2) * [0] + ['wakeup'] if llg > 1 else [0])
    return lstPOI

# 📚 Apply the event extraction function and store the results in the "event" column
test["event"] = get_event(test)


# # 📄 submission file

# In[8]:


# 📄 Create a sample_submission DataFrame by selecting relevant columns and resetting the index
sample_submission = test.loc[test["event"] != 0][["series_id", "step", "event", "score"]].copy().reset_index(drop=True).reset_index(names="row_id")

# 💾 Save the sample_submission DataFrame to a CSV file
sample_submission.to_csv('submission.csv', index=False)

