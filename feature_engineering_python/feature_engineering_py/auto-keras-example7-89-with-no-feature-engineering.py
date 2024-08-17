#!/usr/bin/env python
# coding: utf-8

# # Lets Auto-Keras
# 
# Hello everyone, in this notebook I will show you how to use auto-keras for TPS August.

# Import the Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Install the auto-keras package

# In[ ]:


get_ipython().system('pip install autokeras')


# Load the data

# In[ ]:


import tqdm
from sklearn.metrics import classification_report

data = pd.read_csv('/kaggle/input/tabular-playground-series-aug-2021/train.csv')


# ### Get X and y.
# 
# In the actual submission there is a bit of feature engineering involved, howver since I am only demonstrating the power of auto-keras package, I will not be doing any feature engineering here.

# In[ ]:


y = data.loss.values
X = data.drop(['loss'], axis = 1).values

print("train_shape",X.shape)
print("label shape", y.shape)


# In[ ]:


import tensorflow as tf

import autokeras as ak


# ## Define autokeras model
# 
# Here we will try the Structured data Regressor, since the data is in table format.

# In[ ]:


# Initialize the structured data regressor.
reg = ak.StructuredDataRegressor(
    loss="mean_squared_error",
    overwrite=True,
    objective="val_loss",
    max_trials=10,
    project_name="tps_august"
)  # It tries 10 different models.


reg.fit(
    # The path to the train.csv file.
    '/kaggle/input/tabular-playground-series-aug-2021/train.csv',
    # The name of the label column.
    "loss",
    epochs=100,
)


# ### Make predictions

# In[ ]:


predicted_y = reg.predict('../input/tabular-playground-series-aug-2021/test.csv')


# ### Write to submission file

# In[ ]:


sub = pd.read_csv('../input/tabular-playground-series-aug-2021/sample_submission.csv')
sub.loss = predicted_y
sub.to_csv('sub.csv')


# # Score: 7.99319
# 
# ### We achieved a score of 7.99319 which not very bad considering there was no feature engineering involved at all and we did not try to come up with a sophisticated method to make predictions.
# 
# ### Currently the top score is 7.84, so the auto-keras model is 0.15 behind it only.
# 
# ### With some feature engineering and hyperparameter optimisation while defining the model, I managed to achieve a score of 7.849, so it is clear that auto keras shows a lot of potential when it comes to structured data
# 
# ## Do upvote if you like it, and do let me know if you want to see more automl methods
