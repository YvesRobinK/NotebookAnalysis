#!/usr/bin/env python
# coding: utf-8

# ## Predicting student performance with datatable LinearModel
# 
# [datatable](https://datatable.readthedocs.io/) is a Python package for manipulating data frames. It is close in spirit to pandas, however, specific emphasis is put on speed and big data support. A limited number of machine learning models and tools are also available in the package.
# 
# In this notebook we will focus on the newly developed datatable [LinearModel](https://datatable.readthedocs.io/en/latest/api/models/linear_model.html) to see how well it can predict student performance from game play. We will also use datatable to read the competition data, do data munging and feature engineering.
# 
# There are already many great baselines published for this competition, that involve much more sophisticated models. From the other hand, there are pretty tough restrictions on the computational resources. This makes us to believe that `LinearModel` may also have some potential, as it is fast, lightweight and has high interpretability. So let's give it a try!
# 
# *Spoiler: after just six seconds of fitting, the model achieves the same F1 score as much more sophisticated baselines.*
# 
# ## Contents
#  - [Install datatable and read in the competition data](#install) 
#  - [Feature engineering](#features)
#  - [Fitting the model](#fit)
#  - [Determining the optimal threshold](#threshold)
#  - [Predicting and submitting](#predict)
#  - [Conclusions](#conclusions)

# <a id="install"></a>
# ## Install datatable and read in the competition data
# 
# First, we install the [latest datatable wheels](https://www.kaggle.com/datasets/kononenko/datatable), so that the package could be used in a no-internet code competition. 
# 
# Second, we read the training data and the labels with fully parallel [datatable.fread()](https://datatable.readthedocs.io/en/latest/api/dt/fread.html) function. Unlike with pandas, there is no need to manually adjust any of the column types. The resulting data frames fit into 8Gb of RAM out of the box.

# In[1]:


get_ipython().run_cell_magic('time', '', '# Install the latest datatable wheel\n!pip install /kaggle/input/datatable/datatable-1.1.0a2240-cp310-cp310-manylinux_2_17_x86_64.whl\nfrom datatable import dt, by, f, g\n\n# Read the training data\nDT_train_full = dt.fread("/kaggle/input/predict-student-performance-from-game-play/train.csv")\n\n# Read labels, then parse session ids and question numbers\nDT_labels = dt.Frame("/kaggle/input/predict-student-performance-from-game-play/train_labels.csv")\nDT_labels = DT_labels[:, {\n                          "session_id" : f.session_id[:17].as_type(dt.int64),\n                          "q" : f.session_id[19:].as_type(dt.int8),\n                          "correct" : f.correct\n                     }]\n')


# In[2]:


DT_train_full


# In[3]:


DT_labels


# <a id="features"></a>
# ## Feature engineering
# 
# In this notebook we will stick to a limited number of features, namely
# - number of events per a session;
# - session time;
# - number of unique elements in each of the categorical columns.

# In[4]:


# Feature engineering on `DT` columns
def get_features(DT): 
    # From the numeric columns we create the following two features: 
    # - number of events per session, i.e. `session_id.count()`;
    # - session time, i.e. `elapsed_time.max()`.
    # To reduce skewness, we also apply log transformation to these two features.
    features = {
                "session_id_log_count" : dt.math.log(1 + f.session_id.count()),
                "elapsed_time_log_max" : dt.math.log(1 + f.elapsed_time.max()),
               }

    # From the categorical columns we create the corresponding "nunique" features
    CAT_COLS = ("event_name", "name", "text", "fqid", "room_fqid", "text_fqid")
    for col in CAT_COLS:
        features[col + "_nunique"] = f[col].nunique()
    
    # The above aggregations are done by `session_id` and `level_group` columns   
    return DT[: , features, by("session_id", "level_group")]

get_ipython().run_line_magic('time', 'DT_train = get_features(DT_train_full)')

# Make sure the number of sessions per a level group in the training data
# is the same as the number of sessions per question in labels.
print("Training data consistent:", DT_train.nrows//3 == DT_labels.nrows//18)


# In[5]:


DT_train


# <a id="fit"></a>
# ## Fitting the model
# 
# For each question we create one `LinearModel`, i.e. `18` models in total. We also standardize all the training data to improve performance of these models.

# In[6]:


get_ipython().run_cell_magic('time', '', 'from datatable.models import LinearModel\n\n# Correspondance of level groups and question numbers.\nlevel_groups = {"0-4"   : (1, 4), \n                "5-12"  : (4, 14), \n                "13-22" : (14, 19)}\n\n# Get a level group the q-th question corresponds to\ndef get_level_group(q):\n    for level_group, (q_from, q_to) in level_groups.items():\n        if q >= q_from and q < q_to:\n            return level_group\n    \n# Model parameters\neta0 = 0.0005 # Learning rate\nnepochs = 10  # Number of training epochs \n\nmodels = []\ntrain_stats = []\nDT_train_qs = []\nDT_labels_qs = []\n\n# Fit one linear model per question\nfor q in range(1, 19):\n    print("Fitting model for question %d..." % q)\n    model = LinearModel(eta0=eta0, nepochs=nepochs)\n    \n    # Preparing training data.\n    # For question `q` we determine the corresponding `level_group`,\n    # and then use training data for this `level_group` only.\n    level_group_q = get_level_group(q)\n    DT_train_q = DT_train[f.level_group == level_group_q, :]\n\n    # Preparing labels.\n    # First, we select labels for question `q`. To make sure\n    # these labels go in the same order as the training data,\n    # we join `DT_train_q` and `DT_labels_q` on `session_id`.\n    DT_labels_q = DT_labels[f.q == q, :]\n    DT_labels_q.key = ["session_id"]\n    DT_labels_q = DT_train_q[:, g.correct, dt.join(DT_labels_q)]\n    DT_labels_qs.append(DT_labels_q)\n    \n    # Finally, we can remove first two columns, i.e. `session_id` \n    # and `level_group`, from the training data.\n    DT_train_q = DT_train_q[:, 2:] \n\n    # Calculate mean and standard deviation for each feature.\n    # Then, standardize training data for each question to improve\n    # performance of the linear model.\n    train_stats.append({"mean" : DT_train_q.mean(), \n                        "sd"   : DT_train_q.sd()})\n    f_stand = (f[:] - train_stats[q - 1]["mean"]) / train_stats[q - 1]["sd"]\n    DT_train_qs.append(DT_train_q[:, f_stand])\n    \n    # Fit the q-th model and store it\n    model.fit(DT_train_qs[q - 1], DT_labels_qs[q - 1])\n    models.append(model)\n')


# <a id="threshold"></a>
# ## Determining the optimal threshold
# 
# We are going to use the fitted models to make predictions. In order to convert predicted probabilities into labels, we need to determine the optimal threshold, i.e. the one which results in the largest F1 score on the training data.

# In[7]:


import numpy as np
from sklearn.metrics import f1_score

# Predict on the training data for each question
y_true = []
DT_preds = dt.Frame()
for q in range(1, 19):
    DT_pred = models[q - 1].predict(DT_train_qs[q - 1])
    DT_preds.rbind(DT_pred)
    y_true.extend(DT_labels_qs[q - 1].to_list()[0])

# Go through the reasonable thresholds to determine the one,
# that gives the largest F1 score
f1_max = 0
threshold_max = 0
for threshold in np.arange(0.6, 0.7, 0.01):
    y_pred = DT_preds[:, f["True"] > threshold].to_list()[0]
    f1 = f1_score(y_true, y_pred, average="macro")
    print("Threshold: %f -> F1 score: %f" % (threshold, f1))
    if f1 > f1_max:
        f1_max = f1
        threshold_max = threshold
        
print("Optimal threshold: %f -> F1 score: %f" % (threshold_max, f1_max))


# <a id="predict"></a>
# ## Predicting and submitting
# 
# To make predictions, we standardize the test data with means and standard deviations saved at the fitting step. To convert predicted probabilities into classes, we use `threshold_max`, that demonstrated the largest F1 score on the training data.

# In[8]:


get_ipython().run_cell_magic('time', '', 'import jo_wilder_310\n\njo_wilder_310.make_env.__called__ = False\nenv = jo_wilder_310.make_env()\niter_test = env.iter_test()\n\nfor (PD_test, PD_submission) in iter_test:\n    # Create datatable frame from pandas\n    DT_test = dt.Frame(PD_test)\n    \n    # Do feature engineering for the test data\n    DT_test = get_features(DT_test)\n    \n    # Get the level group information. We are supposed to predict\n    # for one level group at a time.\n    level_group = DT_test[0, "level_group"]\n    (q_from, q_to) = level_groups[level_group]\n    \n    # Remove `session_id` and `level_group` columns from the test data\n    DT_test = DT_test[:, 2:]\n\n    print("\\nPredicting for questions %d-%d" % (q_from, q_to - 1))\n    for q in range(q_from, q_to):\n        # Predict for the standardized test data\n        f_stand = (f[:] - train_stats[q - 1]["mean"]) / train_stats[q - 1]["sd"]\n        DT_p = models[q - 1].predict(DT_test[:, f_stand])\n        \n        # Convert probabilities to labels by using F1-optimal threshold\n        DT_p = DT_p[:, (f["True"] > threshold_max).as_type(dt.int8)]\n        \n        # Set up predictions for the q-th question in the submission frame\n        mask_q = PD_submission.session_id.str.contains(f\'q{q}\')\n        PD_submission.loc[mask_q, \'correct\'] = DT_p.to_list()\n    \n    print(PD_submission)\n    env.predict(PD_submission)     \n')


# <a id="conclusions"></a>
# ## Conclusions
# 
# Despite less than ten seconds of training time, `LinearModel` achieves F1 score of `0.676`, that is similar to much more sophisticated baselines ([TensorFlow Decision Forests](https://www.kaggle.com/code/gusthema/student-performance-w-tensorflow-decision-forests) `0.672`, [XGBoost Baseline](https://www.kaggle.com/code/cdeotte/xgboost-baseline-0-680) `0.677`, etc.). 
# 
# Clearly, the model could be further improved by exploring other numeric features and paying more attention to categoricals. However, it definitely has a lot of potential when it comes to efficiency and limited computational resources.
