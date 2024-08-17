#!/usr/bin/env python
# coding: utf-8

# # MLJAR AutoML 
# 
# MLJAR is an Automated Machine Learning framework. It is available as Python package with code at GitHub: https://github.com/mljar/mljar-supervised
# 
# The MLJAR AutoML can work in several modes:
# - Explain - ideal for initial data exploration
# - Perform - perfect for production-level ML systems
# - Compete - mode for ML competitions under restricted time budget. By the default, it performs advanced feature engineering like golden features search, kmeans features, feature selection. It does model stacking.
# - Optuna - uses Optuna to highly tune algorithms: Random Forest, Extra Trees, Xgboost, LightGBM, CatBoost, Neural Network. Each algorithm is tuned with `Optuna` hperparameters framework with selected time budget (controlled with `optuna_time_budget`). By the default feature engineering is not enabled (you need to manually swtich it on, in AutoML() parameter).
# 
# 
# ## Explain
# 
# The example useage of `Explain` with `MLJAR`:
# 
# ```python
# 
# automl = AutoML(mode="Explain")
# automl.fit(X, y)
# ```
# 
# The best choice to get initial information about your data. This mode will produce a lot of explanations for your data. All details can be viewed in the Notebook by calling the `automl.report()` method.
# 
# 
# ## Compete
# 
# The example useage of `Compete` with `MLJAR`:
# 
# ```python
# 
# automl = AutoML(mode="Compete",
#                 total_time_limit=8*3600)
# automl.fit(X, y)
# ```
# 
# That's it. It will train: Random Forest, Extra Trees, Xgboost, LightGBM, CatBoost, Neural Network, Ensemble, and stack all the models. Feature engineering will be applied (if enough training time). 
# 
# 
# ## Optuna
# 
# The example useage of `Optuna` with `MLJAR`:
# 
# ```python
# 
# automl = AutoML(mode="Optuna", 
#                 optuna_time_budget=1800, 
#                 optuna_init_params={}, 
#                 algorithms=["LightGBM", "Xgboost", "Extra Trees"], 
#                 total_time_limit=24*3600)
# automl.fit(X, y)
# ```
# 
# Description of parameters:
# - `optuna_time_budget` - time budget for `Optuna` to tune each algorithm,
# - `optuna_init_params` - if you have precomputed parameters for `Optuna` they can be passed here, then for already optimized models `Optuna` will not be used.
# - `algorithms` - the algorithms that we will check,
# - `total_time_limit` - the total time limit for AutoML training.
# 
# (In the `Optuna` mode, only first fold is used for model tuning.)
# 
# ---
# 
# MLJAR GitHub: https://github.com/mljar/mljar-supervised
# 
# Optuna GitHub: https://github.com/optuna/optuna
# 
# <img src="https://raw.githubusercontent.com/mljar/visual-identity/main/media/kaggle_banner_white.png" style="width: 70%;"/>

# In[1]:


get_ipython().system('pip install -q -U git+https://github.com/mljar/mljar-supervised.git@master')


# In[2]:


import numpy as np
import pandas as pd
from supervised.automl import AutoML # mljar-supervised


# In[3]:


train = pd.read_csv("../input/tabular-playground-series-apr-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2021/test.csv")


# In[4]:


train.shape, test.shape


# In[5]:


train.head()


# In[6]:


x_cols = train.columns[2:].tolist()
y_col = train.columns[1]


# In[7]:


automl = AutoML(
    mode="Compete", 
    eval_metric="f1",
    total_time_limit=3600,
    features_selection=False # switch off feature selection
)
automl.fit(train[x_cols], train[y_col])


# In[8]:


preds = automl.predict(test[x_cols])


# In[9]:


preds


# In[10]:


submission = pd.DataFrame({'PassengerId':test.iloc[:,0], 'Survived': preds})
submission.to_csv('1_submission.csv', index=False)


# In[11]:


automl.report()


# In[ ]:





# In[ ]:




