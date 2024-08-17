#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
df


# In[2]:


get_ipython().system('pip install lofo-importance')


# ![alt text](https://raw.githubusercontent.com/aerdem4/lofo-importance/master/docs/lofo_logo.png)
# 
# LOFO (Leave One Feature Out) Importance calculates the importances of a set of features based on a metric of choice, for a model of choice, by iteratively removing each feature from the set, and evaluating the performance of the model, with a validation scheme of choice, based on the chosen metric.
# 
# LOFO first evaluates the performance of the model with all the input features included, then iteratively removes one feature at a time, retrains the model, and evaluates its performance on a validation set. The mean and standard deviation (across the folds) of the importance of each feature is then reported.
# 
# If a model is not passed as an argument to LOFO Importance, it will run LightGBM as a default model.
# 
# ## Install
# 
# LOFO Importance can be installed using
# 
# ```
# pip install lofo-importance
# ```
# 
# ## Advantages of LOFO Importance
# 
# LOFO has several advantages compared to other importance types:
# 
# * It does not favor granular features
# * It generalises well to unseen test sets
# * It is model agnostic
# * It gives negative importance to features that hurt performance upon inclusion
# * It can group the features. Especially useful for high dimensional features like TFIDF or OHE features.
# * It can automatically group highly correlated features to avoid underestimating their importance.

# ## Feature Engineering

# In[3]:


df["wait_time"] = df["down_time"] - df.groupby("id")["up_time"].shift()
df["activity"] = df["activity"].apply(lambda x: "Move" if x.startswith("Move") else x)

activity_ohe_df = pd.get_dummies(df["activity"])
activity_cols = list(activity_ohe_df.columns)

df = pd.concat([df, activity_ohe_df], axis=1)


# In[4]:


agg = {"event_id": ["max"], 
       "action_time": ["mean", 'std', "max", "min"], 
       "wait_time": ["mean", "std", "max", "min"],
       "word_count": ["last"]
      }

for col in activity_cols:
    agg[col] = ["mean"]


agg_df = df.groupby("id").agg(agg).reset_index()
agg_df.columns = [f"{col[0]}_{col[1]}".strip("_") for col in agg_df.columns]


# In[5]:


agg_df = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv").merge(agg_df, on="id")
agg_df.head()


# ## Define LOFO Dataset

# In[6]:


import lofo

target = "score"
features = ["event_id_max", 
            "action_time_mean", "action_time_std", "action_time_max", "action_time_min", 
            "wait_time_mean", "wait_time_std", "wait_time_min", "wait_time_max",
            "word_count_last"
           ]

feature_groups = {"activity_mean": agg_df[[f"{col}_mean" for col in activity_cols]].values}

ds = lofo.Dataset(agg_df, target=target, features=features, feature_groups=feature_groups,
                  auto_group_threshold=0.7)


# ## Run LOFO

# In[7]:


lofo_imp = lofo.LOFOImportance(ds, cv=5, scoring="neg_root_mean_squared_error")
imp_df = lofo_imp.get_importance()


# ## Feature Importance

# In[8]:


lofo.plot_importance(imp_df, figsize=(12, 6))


# In[ ]:




