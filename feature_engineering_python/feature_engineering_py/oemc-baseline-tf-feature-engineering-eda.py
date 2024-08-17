#!/usr/bin/env python
# coding: utf-8

# ***This is work in progress. Expect more detailed EDA and feature engineering in future versions and also training and inference.***
# 
# ***Version 3 : Includes baseline tensorflow submission***

# In[30]:


#Basic imports required for EDA
import numpy as np                    
import pandas as pd                   
import matplotlib.pyplot as plt        
import seaborn as sns 


# In[31]:


#setting path for training set, test set and submission
train = pd.read_csv(r"/kaggle/input/oemc-hackathon-global-fapar-modeling/train.csv")
test  = pd.read_csv(r"/kaggle/input/oemc-hackathon-global-fapar-modeling/test.csv")
submission = pd.read_csv(r"/kaggle/input/oemc-hackathon-global-fapar-modeling/sample_submission.csv")


# In[32]:


#taking a look at training and test set
train.head()


# In[33]:


test.head()


# In[34]:


train.columns


# In[35]:


#Looking into missing values in the data
train.isna().sum()


# In[36]:


train.info()


# **Plotting some heatmaps to check for the correlations**

# In[37]:


sns.heatmap(train.iloc[:,[1,2,3]].corr(),annot=True,fmt='0.2f')


# In[38]:


sns.heatmap(train.iloc[:,[1,2,3,4,5,6,7,8,9]].corr(),annot=True,fmt='0.2f')


# In[39]:


sns.heatmap(train.iloc[:,[1,2,3,10,11,12,13,14,15]].corr(),annot=True,fmt='0.2f')


# **It can be noticed that modis_red and modis_blue have correlation with station and month. However, there is no negative correlation of modis_evi and modis_ndvi. But modis_evi and modis_ndvi are highly correlated with Fapar. Interesting!**

# In[53]:


import tensorflow as tf

# Extract the features and labels
features = train.drop("fapar", axis=1)
labels = train["fapar"]

# Create a dense model with two hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer="adam", loss="mae")

# Train the model
model.fit(features, labels, epochs=100)

# Predict the FAPAR values for the test data
predictions = model.predict(test)

# Load the predictions into a Pandas DataFrame
predictions_df = pd.DataFrame(predictions, columns=["fapar"])

with open("predictions.csv", "w") as f:
    f.write("sample_id,fapar\n")
    for i, prediction in enumerate(predictions_df["fapar"]):
        f.write(f"{i},{format(prediction, '.4f')}\n")

