#!/usr/bin/env python
# coding: utf-8

# <div>
# <h1 style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;text-align:center" >Autogluon Quick Start</h1>
# </div>

# # Table of Contents
# <a id="toc"></a>
# - [1. Introduction to Autogluon](#1)
#     - [1.1 How it works](#1.1)
#     - [1.2 Application scenarios](#1.2)
# - [2. Install and Import libraries](#2)
# - [3. Data Loading and Pre-Processing](#3)
#     - [3.1 Data Loading](#3.1)
#     - [3.2 Preperation](#3.2)
#     - [3.3 Imputing Missing Values](#3.3)
#     - [3.4 Encoding](#3.4)
# - [4. Build Model and Predict](#4)
#     - [4.1 Training Model](#4.1)
#     - [4.2 Predict](#4.2)
# - [5. Submission](#5)   

# <a id="1"></a>
# # **<center><span style="background-color:#1e2150;padding:20px;border-radius:10px;border:3px solid #e3b4b9;color:#eadde8;text-align:center">What is Autogluon?</span></center>**

# <div style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;" >
#     
# -  In recent years, machine learning has made significant breakthroughs in various fields, and is increasingly being applied in life sciences and medicine. However, trying to actually build a model is still time-consuming and takes quite some time to learn.
# </div>

# <div style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;" >
#     
# -  To address this prominent problem, Automatic Machine Learning (AutoML) was created, the process of automating the end-to-end process of applying machine learning to real-world problems. autoML makes machine learning truly possible, even for people with no expertise in the field.
#    
# </div>

# <div style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;" >
#     
# -  AutoGluon, an AutoML solution that Amazon open sourced on GitHub, builds on the Gluon deep learning library that Amazon launched three years ago in conjunction with Microsoft to automate decisions by automatically adjusting choices within constraints to find the optimal model using available resources.
# </div>

# <div align="center">
# <img src="https://www.bizety.com/wp-content/uploads/2020/06/AutoGluon.png" alt="">
# </div>

# <div style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;" >
# AutoGluon is convenient for:
#     
# - Rapid construction of deep learning prototype solutions for data with a few lines of code.
#     
# - Using automatic hyperparameter fine-tuning, model selection/architecture search and data processing.
#     
# - Automating the use of deep learning SOTA methods without expert knowledge.
#     
# - Easily enhance existing custom models and data pipelines, or customize AutoGluon based on use cases.
# </div>

# <div style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;" >
#        
# One most important AutoML features in AutoGluon is called neural architecture search. These tools optimize the weights, structure, and hyperparameters of an ML model’s algorithmic “neurons” to ensure the accuracy, speed, and efficiency of neural nets in performing data-driven inferences. Neural Architecture Search allows AI developers to automate the optimization of models for high-performance inferencing on various hardware platforms.
# 
# AutoGluon automatically generates a high-performance ML model from Python code. It taps into available reinforcement learning (RL) algorithms and computes resources to search for the best-fit neural-network architecture for the target environment.
# </div>

# <div align="center">
# <img src="https://d2908q01vomqb2.cloudfront.net/ca3512f4dfa95a03169c5a670a4c91a19b3077b4/2020/03/30/autogluon_f2.png" alt="">
# </div>

# <a id="1.1"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150">Here’s how this model works</span>**

# <div style="padding:20px;border-radius:10px;border:3px solid #e3b4b9;" >
# 
# - Base layer: Trains individual base models, such as Random Forests, CatBoost boosted trees, LightGBM boosted trees, Extremely Randomized trees, etc.
#     
# - Concat layer: Base layer’s output is concatenated with the input features.
#     
# - Stacker layer: Trains multiple stacker models on the concat layer output. This layer re-uses the exact same models in the base layer. Stacker models can look at the input dataset because input features are concatenated with the output of the base layer.
#     
# - Weighting layer: Implements an ensemble selection approach in which stacker models are introduced into a new ensemble.
#     
# AutoGluon-Tabular performs k-fold cross-validation to ensure that every learner can see the entire dataset.
# </div>

# https://www.bizety.com/2020/06/16/open-source-automl-tools-autogluon-transmogrifai-auto-sklearn-and-nni/

# <div align="center">
# <img src="https://th.bing.com/th/id/R.33796bf9e531446f95ac289da4ed2966?rik=w2G6gBsitNWs7g&riu=http%3a%2f%2f3qt70435i1bu2glt7n1697lt4mg-wpengine.netdna-ssl.com%2fwp-content%2fuploads%2f2020%2f06%2fEnsembles.png&ehk=kIgtTzkdaS4Nk8PKXmGHR%2fjPTc%2bRu792Gk1i%2fSE1o%2bU%3d&risl=&pid=ImgRaw&r=0" alt="">
# </div>

# <a id="1.2"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Application scenarios</span>**
# 
# <div style="padding:20px;border-radius:15px;border:3px solid #e3b4b9;" >
#     
# - Machine learning: implementation of prediction problems in machine learning.
# 
# - Image classification: identifying the main objects in an image.
# 
# - Object detection: detecting multiple objects with the help of bounding boxes in images.
# 
# - Text classification: making predictions based on text content.
# </div>

# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="2"></a>
# # **<center><span style="background-color:#1e2150;padding:20px;border-radius:10px;border:3px solid #e3b4b9;color:#eadde8;text-align:center">Install and import Libraries</span></center>**

# In[1]:


# Run this if autogluon is not already installed

get_ipython().system('pip install "mxnet<2.0.0"')
get_ipython().system('pip install autogluon')


# In[2]:


# Importing the necessary libraries

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="3"></a>
# # **<center><span style="background-color:#1e2150;padding:20px;border-radius:10px;border:3px solid #e3b4b9;color:#eadde8;text-align:center">Data reading and Pre-Processing</span></center>**

# <a id="3.1"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Data reading</span>**

# In[3]:


# Reading data

train = TabularDataset("../input/spaceship-titanic/train.csv")
test = TabularDataset("../input/spaceship-titanic/test.csv")


# <a id="3.2"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Preperation</span>**

# In[4]:


# Set the label to be predicted and the id column to be deleted

id, label = "PassengerId", "Transported"
eval_metric = "accuracy"

df = pd.concat([train,test],axis=0)
num_columns = []
for col in train:
    if train[col].dtypes != "object" and col != label:
        num_columns.append(col)


# <a id="3.3"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Imputing Missing Values</span>**

# In[5]:


# https://www.kaggle.com/code/odins0n/spaceship-titanic-eda-27-different-models
STRATEGY = 'median'
imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]
imputer = SimpleImputer(strategy=STRATEGY)

df[imputer_cols] = imputer.fit_transform(df[imputer_cols])
df["HomePlanet"].fillna('Z', inplace=True)


# <a id="3.4"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Encoding </span>**

# In[6]:


label_cols = ["HomePlanet", "CryoSleep","Cabin", "Destination" ,"VIP"]
for col in label_cols:
    df[col] = df[col].astype(str)
    df[col] = LabelEncoder().fit_transform(df[col])


# In[7]:


ss = StandardScaler()
df[num_columns] = ss.fit_transform(df[num_columns])


# In[8]:


train = df[:len(train)]
test = df[len(train):]


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4"></a>
# # **<center><span style="background-color:#1e2150;padding:20px;border-radius:10px;border:3px solid #e3b4b9;color:#eadde8;text-align:center">Build Model and Predict</span></center>**

# <a id="4.1"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Training Model</span>**

# In[9]:


# Training models

models = TabularPredictor(label=label, eval_metric=eval_metric).fit(train.drop(columns=[id]))


# <a id="4.2"></a>
# # **<span style="padding:10px;border-radius:10px;border:3px;color:#1e2150;">Predict</span>**

# In[10]:


# Predict

pred = models.predict(test.drop(columns=[id]))


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="5"></a>
# # **<center><span style="background-color:#1e2150;padding:20px;border-radius:10px;border:3px solid #e3b4b9;color:#eadde8;text-align:center">Submission</span></center>**

# In[11]:


# Submitting the file

submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")

submission[label] = pred
submission.to_csv("submission.csv", index=False)


# <p style="background-color:#1e2150;font-size:25px;padding:15px;border-radius:10px;color:#eadde8;text-align:center">
# Thank you for watching to the end, if it helps you, please give an upvote！
# </p>

# ## 🔥 Automated EDA Tools (Part 1) 🖋️
# https://www.kaggle.com/code/mozattt/automated-eda-tools-part-1
# 
# ## 🔥 Automated EDA Tools (Part 2) 🖋️
# https://www.kaggle.com/code/mozattt/automated-eda-tools-part-2
# 
# ## 🔥Share some feature engineering libraries:
# https://www.kaggle.com/code/mozattt/share-some-feature-engineering-libraries
# 
# ## 📊ML_Marathon｜Exploratory analysis of data
# https://www.kaggle.com/code/mozattt/ml-marathon-exploratory-analysis-of-data
