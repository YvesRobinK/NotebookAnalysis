#!/usr/bin/env python
# coding: utf-8

# # Career Exploration Kaggle Competition: Rain in Australia Prediction
# 
# ### Hosted by and maintained by the [Students Association of Applied Statistics (SAAS)](https://saas.berkeley.edu).  Authored by Derek Cai(dcai@berkeley.edu).

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd 


# ## Loading Data

# In[2]:


X_train = pd.read_csv("../input/saas-2021-spring-cx-kaggle-compeition/train_features.csv")
X_test = pd.read_csv("../input/saas-2021-spring-cx-kaggle-compeition/test_features.csv")
y_train = pd.read_csv("../input/saas-2021-spring-cx-kaggle-compeition/train_targets.csv")
del y_train['Id']
sample_submission = pd.read_csv("../input/saas-2021-spring-cx-kaggle-compeition/sample_submission.csv")


# ## Feature Engineering

# In our Visualization + Data Cleaning HW, you have explored and got familiar the dataset. Now is the time to do some feature engineering! 

# In[3]:


#as mentioned in lecture, when doing feature engineering, it's better to merge train and test datasets first and do operations on the entire merged dataset
full_data = pd.concat([X_train, X_test]).reset_index(drop=True)


# In[4]:


#checking if the merged dataset has the correct number of rows
assert full_data.shape[0] == X_train.shape[0]+X_test.shape[0]


# In[5]:


full_data.columns


# The above shows all the column names of the training dataset. Do you think all columns are useful in doing rain prediction? For instance, is knowing the date really going to help us?

# ### Dropping Features
# Not all features are useful. Drop some features of full_data if you want!
# Resource: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html

# In[6]:


# Your Code Here


# In[7]:


# original full_data has 22 columns, after you drop some columns, the below expression will return a number less than 23
len(full_data.columns)


# ### Fill in the Missing Values
# In Visualization + Data Cleaning HW, we have explored techniques of dealing with missing values. Perform the same techniques on all columns of full_data that contain missing values. 

# In[8]:


# below expression shows the number of missing values in each column
full_data.isna().sum()


# In[9]:


# Your code here


# ### One Hot Encoding
# One-hot encode categorical columns

# In[10]:


# Your code here


# ### Optional: Dimensionality Reduction
# When the data has high dimensions(a lot of columns), it is very useful to use PCA to lower the dimension of the data during feature engineering. 
# Since we only have around 20 features, this is not necessary. But it could potentially help with your kaggle score.
# PS: PCA is designed for continuous variables, so maybe you should try ignore categorical columns for PCA.
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

# In[11]:


# optional
from sklearn.decomposition import PCA
# Your code here


# ### Feel Free to do more feature engineering on df! All the methods listed above are ones to help you get started.

# In[12]:


# Splitting up our engineered df back into training and test
X_train = full_data[:X_train.shape[0]]
X_test = full_data[X_train.shape[0]:]


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# ## Baseline Model
# Congrats! You have feature engineered the train and test dataset such that they can be used to build models!

# Our Kaggle competition uses MAE(Mean Absolute Error) as our metric!

# In[15]:


from sklearn.metrics import mean_absolute_error
def evaluate(y_pred, y_true):
    """Returns the MAE(y_pred, y_true)"""
    return mean_absolute_error(y_true, y_pred)


# In[16]:


# Build a simple random forest model here
# Don't worry if you don't how a random forest works. We will cover that in lecture. 
# The below code serves as a demonstration of how you generally create a model
from sklearn.ensemble import RandomForestClassifier
# Instantiate a model 
clf = RandomForestClassifier(max_depth=1, random_state=0, verbose=1)
# Train the model using our train_features and train_targets
clf.fit(X_train.to_numpy(), np.ravel(y_train.to_numpy()))
# Use the trained model to predict y_train!
predictions = clf.predict(X_test)


# ## Cross Validation
# ![cross-validation-graphic](https://i.stack.imgur.com/1fXzJ.png)

# Task 1:
# In Lecture, we have discussed the cross-validtion scheme. Set up your cross validation below. Perform a 5-fold cross-validation on the rain dataset below. You should use 20% of your training data as your validation data. Print out the accuracy of each of your 5 experiment! Feel free to keep the random forest model or use any models of your choosing. DM the screenshot of your code and the printed out 5 experiment accuracy to Derek! 

# Task2: For people scoring above 0.17 on the leaderboard, beat the score 0.17! For people scoring below 0.17, beat your current score! 

# ## Ensemble

# Create another model that scores decently well and create an ensemble of these 2 models. You can simply average the output predictions of these 2 models or take a weighted average of the output predictions of of these 2 models. Screenshot the code of your second model as well as the final ensemble process to Derek. Your submission should generally be better than your single submission.

# In[ ]:





# In[ ]:





# ## Submission

# In[17]:


assert predictions.shape[0] == 25124


# In[18]:


sample_submission['RainTomorrow'] = predictions


# In[19]:


sample_submission.to_csv("submission.csv", index=False)


# ### Note: submission.csv can be found within the /kaggle/working file on the right side of your screen. Download the csv file and use it for submission!
