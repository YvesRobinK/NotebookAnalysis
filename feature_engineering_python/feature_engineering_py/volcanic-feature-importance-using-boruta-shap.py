#!/usr/bin/env python
# coding: utf-8

# # Volcanic feature importance using Boruta-SHAP
# In this notebook we shall produce a selection of the most important features of the [INGV - Volcanic Eruption Prediction](https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe) data using the [Boruta-SHAP](https://www.kaggle.com/carlmcbrideellis/feature-selection-using-the-borutashap-package) package.
# For the input I use the `train.csv` produced by the excellent notebook ["INGV Volcanic Eruption Prediction - LGBM Baseline"](https://www.kaggle.com/ajcostarino/ingv-volcanic-eruption-prediction-lgbm-baseline) written by [Adam James](https://www.kaggle.com/ajcostarino). I shall write the results of the feature selection to the file `selected_features.csv`.

# In[1]:


import numpy   as np
import pandas  as pd
pd.set_option('display.max_columns', None)
get_ipython().system('pip install BorutaShap')


# In[2]:


train   = pd.read_csv('../input/the-volcano-and-the-regularized-greedy-forest/volcano_train.csv')
X_train = train.drop(["segment_id","time_to_eruption"],axis=1)
y_train = train["time_to_eruption"]

from xgboost import XGBRegressor
model = XGBRegressor()

from BorutaShap import BorutaShap
Feature_Selector = BorutaShap(model=model,importance_measure='shap', classification=False)
Feature_Selector.fit(X=X_train, y=y_train, n_trials=35, random_state=0);


# Produce a box-plot of the accepted features

# In[3]:


Feature_Selector.plot(which_features='accepted', figsize=(20,12))


# Return a subset of the original data with the selected features

# In[4]:


selected_features = Feature_Selector.Subset()
selected_features


# write out a `selected_features.csv` file

# In[5]:


selected_features.to_csv('selected_features.csv',index=False)


# ### Produce a `submission.csv` using the RGF
# For completeness we shall produce a `submission.csv`, here using the [Regularized Greedy Forest](https://www.kaggle.com/carlmcbrideellis/introduction-to-the-regularized-greedy-forest) for the estimator:

# In[6]:


test   = pd.read_csv('../input/the-volcano-and-the-regularized-greedy-forest/volcano_test.csv')
X_train          = selected_features
selected_columns = selected_features.columns
X_test           = test[selected_columns]

from rgf.sklearn import RGFRegressor
regressor = RGFRegressor(max_leaf=10000, 
                         algorithm="RGF_Sib", 
                         test_interval=100, 
                         loss="LS",
                         verbose=False)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

sample = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
sample.iloc[:,1:] = predictions
sample.to_csv('submission.csv',index=False)

