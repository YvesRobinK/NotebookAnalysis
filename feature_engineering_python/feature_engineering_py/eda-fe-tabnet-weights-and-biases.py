#!/usr/bin/env python
# coding: utf-8

# ![](https://drive.google.com/uc?id=1ubiwsZtL3GcfnrMhJI_6Ls_73qrnRwPH)

#  # <h1 style='background:#F7B2B0; border:0; color:black'><center>Google Brain - Ventilator Pressure Prediction</center></h1> 

# What do doctors do when a patient has trouble breathing? They use a ventilator to pump oxygen into a sedated patient's lungs via a tube in the windpipe. But mechanical ventilation is a clinician-intensive procedure, a limitation that was prominently on display during the early days of the COVID-19 pandemic. At the same time, developing new methods for controlling mechanical ventilators is prohibitively expensive, even before reaching clinical trials. High-quality simulators could reduce this barrier.
# 
# # **<span style="color:#F7B2B0;">Goal</span>**
#  
# The goal is to simulate a ventilator connected to a sedated patient's lung by taking lung attributes compliance and resistance into account.
# 
# # **<span style="color:#F7B2B0;">Data</span>**
# 
# Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.
# 
# **Files**
# > - ``` train.csv``` - the training set
# > - ```test.csv``` - the test set
# > - ```sample_submission.csv``` - a sample submission file in the correct format
# 
# **Columns**
# > - ```id``` - globally-unique time step identifier across an entire file
# > - ```breath_id``` - globally-unique time step for breaths
# > - ```R``` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.
# > - ```C``` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.
# > - ```time_step``` - the actual time stamp.
# > - ```u_in``` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
# > - ```u_out``` - the control input for the exploratory solenoid valve. Either 0 or 1.
# > - ```pressure``` - the airway pressure measured in the respiratory circuit, measured in cmH2O
# 
# # **<span style="color:#F7B2B0;">Evaluation Metric</span>**
# 
# The competition will be scored as the mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored. The score is given by:
# 
#                                         |X-Y|
# 
# where  is the vector of predicted pressure and  is the vector of actual pressures across all breaths in the test set.
# 

# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">
# 
# > I will be integrating W&B for visualizations and logging artifacts!
# > 
# > [Google Brain - Ventilator Pressure Prediction Project on W&B Dashboard](https://wandb.ai/usharengaraju/GoogleBrainVentilatorPressurePrediction)
# > 
# > - To get the API key, create an account in the [website](https://wandb.ai/site) .
# > - Use secrets to use API Keys more securely 
# 
# <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;"> Weights & Biases (W&B) is a set of machine learning tools that helps you build better models faster. <strong>Kaggle competitions require fast-paced model development and evaluation</strong>. There are a lot of components: exploring the training data, training different models, combining trained models in different combinations (ensembling), and so on.</span>
# 
# > <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">⏳ Lots of components = Lots of places to go wrong = Lots of time spent debugging</span>
# 
# <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">W&B can be useful for Kaggle competition with it's lightweight and interoperable tools:</span>
# 
# * <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">Quickly track experiments,<br></span>
# * <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">Version and iterate on datasets, <br></span>
# * <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">Evaluate model performance,<br></span>
# * <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">Reproduce models,<br></span>
# * <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">Visualize results and spot regressions,<br></span>
# * <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">Share findings with colleagues.</span>
# 
# <span style="color: #000508; font-family: Segoe UI; font-size: 1.2em; font-weight: 300;">To learn more about Weights and Biases check out this <strong><a href="https://www.kaggle.com/ayuraj/experiment-tracking-with-weights-and-biases">kernel</a></strong>.</span>
# 
# ![img](https://i.imgur.com/BGgfZj3.png)

# In[1]:


get_ipython().system('pip install pytorch-tabnet')

import os
import gc

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('ggplot')
import seaborn as sns
from scipy import stats

import wandb

from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


df_train = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
df_test  = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
df_sample = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')


# To reduce the running I have taken only 10,000 rows for visualization and modelling . If you wish to take entire dataset , comment out the below code 

# In[3]:


df_train = df_train[:10000]
df_test = df_test[:10000]
common_features = ['breath_id','R','C','time_step','u_in','u_out']
numerical_features = ['time_step','u_in']
categorical_features = ['R','C','u_out','breath_id']


# # **<span style="color:#F7B2B0;">Missing Values</span>**

# In[4]:


plt.figure(figsize = (25,11))
sns.heatmap(df_train.isna().values, cmap = ['#ffd514','#ff355d'], xticklabels=df_train.columns)
plt.title("Missing values in training Data", size=20);


# In[5]:


try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb_api")
    wandb.login(key=secret_value_0)
    anony=None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


# In[6]:


CONFIG = dict(competition = 'VentilatorPressurePrediction',_wandb_kernel = 'tensorgirl')


# # **<span style="color:#F7B2B0;">W & B Artifacts</span>**
# 
# An artifact as a versioned folder of data.Entire datasets can be directly stored as artifacts .
# 
# W&B Artifacts are used for dataset versioning, model versioning . They are also used for tracking dependencies and results across machine learning pipelines.Artifact references can be used to point to data in other systems like S3, GCP, or your own system.
# 
# You can learn more about W&B artifacts [here](https://docs.wandb.ai/guides/artifacts)
# 
# ![](https://drive.google.com/uc?id=1JYSaIMXuEVBheP15xxuaex-32yzxgglV)

# In[7]:


# Save train data to W&B Artifacts
run = wandb.init(project='GoogleBrainVentilatorPressurePrediction', name='training_data', anonymous=anony,config=CONFIG) 
artifact = wandb.Artifact(name='training_data',type='dataset')
artifact.add_file("../input/ventilator-pressure-prediction/train.csv")

wandb.log_artifact(artifact)
wandb.finish()


# Snapshot of the artifacts created  
# 
# ![](https://drive.google.com/uc?id=16ROHOYdW3ewGESfCwewUWW8X3mvNbFKT)

# In[8]:


# basic stats of features
df_train.describe().style.background_gradient(cmap="Pastel1")


# In[9]:


def kdeplot_features(df_train,df_test, feature, title):
    '''Takes a column from the dataframe and plots the distribution (after count).'''
    
    values_train = df_train[feature].to_numpy()
    values_test = df_test[feature].to_numpy()  
     
    plt.figure(figsize = (18, 3))
    
    sns.kdeplot(values_train, color = '#ffd514')
    sns.kdeplot(values_test, color = '#ff355d')
    
    plt.title(title, fontsize=15)
    plt.legend()
    plt.show();
    
    del values_train , values_test
    gc.collect()
    
def countplot_features(df_train, feature, title):
    '''Takes a column from the dataframe and plots the distribution (after count).'''
    
           
    plt.figure(figsize = (10, 5))
    
    sns.countplot(df_train[feature], color = '#ff355d')
        
    plt.title(title, fontsize=15)    
    plt.show();
    
        
def create_wandb_hist(x_data=None, x_name=None, title=None, log=None):
    '''Create and save histogram in W&B Environment.
    x_data: Pandas Series containing x values
    x_name: strings containing axis name
    title: title of the graph
    log: string containing name of log'''
    
    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log : wandb.plot.histogram(table, x_name, title=title)})


# # **<span style="color:#F7B2B0;">Distribution of Features</span>**
# 
# 

# In[10]:


# plot distributions of features
for feature in common_features:
    kdeplot_features(df_train,df_test, feature=feature, title = feature + " distribution")


# Logging plots to W&B dashboard

# In[11]:


# Log Plots to W&B environment
title = "Distribution of features"
run = wandb.init(project='GoogleBrainVentilatorPressurePrediction', name=title,anonymous=anony,config=CONFIG)
for feature in common_features:
    title = "Distribution of "+feature    
    create_wandb_hist(x_data=df_train[feature],x_name=feature , title=title,log="hist")    
wandb.finish()

title = "Countplot Distribution"
run = wandb.init(project='GoogleBrainVentilatorPressurePrediction', name=title,anonymous=anony,config=CONFIG)    
for feature in categorical_features:
    fig = countplot_features(df_train, feature=feature, title = feature + " countplot distribution")
    wandb.log({feature + " countplot distribution": fig})
wandb.finish()


# # **<span style="color:#F7B2B0;">Frequency Distribution of Categorical Features</span>**
# 
# 

# In[12]:


# plot distributions of categorical features
for feature in categorical_features:
    countplot_features(df_train, feature=feature, title = "Frequency of "+ feature)


# # **<span style="color:#F7B2B0;">Distribution of Target Variable - Pressure</span>**

# In[13]:


#histogram
sns.distplot(df_train['pressure'],color = '#ff355d');
fig = plt.figure()
res = stats.probplot(df_train['pressure'], plot=plt)


# # **<span style="color:#F7B2B0;">Numerical Variables Vs Target</span>**

# In[14]:


#Target vs Numerical Features
for feature in numerical_features:
    sns.jointplot(df_train['pressure'],df_train[feature],color = '#ff355d', kind = "kde")     
    plt.show()


# # **<span style="color:#F7B2B0;">Categorical Variables Vs Target</span>**

# In[15]:


#Target vs Categorical Features
for feature in categorical_features:
    sns.boxplot(df_train[feature],df_train['pressure'] ,color = '#ff355d')     
    plt.show()


# # **<span style="color:#F7B2B0;">Analysis for single breath_id</span>**

# The code below is inspired from @vincenttu Brilliant EDA notebook . Kindly upvote his work [here](https://www.kaggle.com/vincenttu/google-vent-eda)

# In[16]:


#code copied from https://www.kaggle.com/vincenttu/google-vent-eda
train_breath_id_2 = df_train[df_train.breath_id == 2] 
train_breath_id_2.style.background_gradient(cmap="Pastel1")


# In[17]:


x = range(80)
plt.figure(figsize = (10, 5))
y1 = train_breath_id_2.u_in
y2 = train_breath_id_2.u_out

plt.xlabel("Time")
plt.ylabel("u_in/u_out range")
sns.lineplot(x, y1, label="u_in",color = '#ffd514')
sns.lineplot(x, y2, label="u_out",color = '#ff355d')
plt.legend()


# In[18]:


plt.figure(figsize = (10, 5))
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.plot(x, train_breath_id_2.pressure.values, label="pressure")
plt.ylabel("pressure/u_in/u_out")
sns.lineplot(x, y1, label="u_in",color = '#ffd514')
sns.lineplot(x, y2, label="u_out",color = '#ff355d')
plt.legend()
plt.legend()


# **Observations:**
# 
# All R and C pairs are the same for any given Breath ID.

# # **<span style="color:#F7B2B0;">Correlation of Features</span>**

# In[19]:


plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.corr().values,linewidths=0.1,vmax=1.0,square=True, cmap="Pastel1", linecolor='white', annot=True)


# In[20]:


sns.pairplot(df_train , height = 2.5 , hue = "R")


#  # **<span style="color:#F7B2B0;">Feature Engineering</span>**

# In[21]:


#Code copied from https://www.kaggle.com/ryanbarretto/tensorflow-lstm-baseline

df_train['u_in_cumsum'] = (df_train['u_in']).groupby(df_train['breath_id']).cumsum()
df_test['u_in_cumsum'] = (df_test['u_in']).groupby(df_test['breath_id']).cumsum()
df_train['u_in_lag'] = df_train['u_in'].shift(2)
df_train = df_train.fillna(0)
df_test['u_in_lag'] = df_test['u_in'].shift(2)
df_test = df_test.fillna(0)


# In[22]:


# Code copied from https://www.kaggle.com/tolgadincer/tensorflow-bidirectional-lstm-0-234

df_train['area'] = df_train['time_step'] * df_train['u_in']
df_train['area'] = df_train.groupby('breath_id')['area'].cumsum()

df_test['area'] = df_test['time_step'] * df_test['u_in']
df_test['area'] = df_test.groupby('breath_id')['area'].cumsum()


# # **<span style="color:#F7B2B0;">TabNet</span>**
# 
# TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning as the learning capacity is used for the most salient features. TabNet outperforms other neural network and decision tree variants on a wide range of non-performance-saturated tabular datasets and yields interpretable feature attributions plus insights into the global model behavior. 
# 
# The main features of TabNet are 
# 
# The main contributions are summarized as:
# 
# 📌 TabNet inputs raw tabular data without any preprocessing
# 
# 📌 TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and better learning as the learning capacity
# 
# 📌 TabNet outperforms or is on par with other tabular learning models on various datasets for classification and regression problems from different domains
# 
# 📌 TabNet shows significant performance improvements by using unsupervised pre-training to predict masked features 
# 
# ![](https://drive.google.com/uc?id=1snKduiQHakIeulnr7jKwt2uQvmv8rDcl)
# 
# [Source](https://arxiv.org/pdf/1908.07442.pdf)
# 
# # **<span style="color:#F7B2B0;">TabNet for Timeseries Data</span>**
# 
# Some resources using TabNet for timeseries data
# 
# [Github](https://github.com/AlbertoCastelo/tabnet-timeseries-spike)
# 
# Short Term Load Forecasting using TabNet - MDPI
# 
# Rainfall Forecast using TabNet - MDPI
# 
# 
# The below explanation is taken from medium article [here](https://towardsdatascience.com/tabnet-e1b979907694)
# 
# # **<span style="color:#F7B2B0;">Steps:</span>**
# 
# Each Step is a block of components. The number of Steps is a hyperparameter option when training the model. Increasing the number of steps will increase the learning capacity of the model, but will also increase training time, memory usage and the chance of overfitting.Each Step gets its own vote in the final classification and these votes are equally weighted. This mimics an ensemble classification.
# 
# # **<span style="color:#F7B2B0;">Feature Transformer:</span>**
# 
# The Feature Transformer is a network which has an architecture of its own.It has multiple layers, some of which are shared across every Step while others are unique to each Step. Each layer contains a fully connected layer, batch normalisation and a Gated Linear Unit activiation.
# 
# Sharing some layers between decision Steps leads to “parameter-efficient and robust learning with high capacity” and that normalization with root 0.5 “helps to stabilize learning by ensuring that the variance throughout does not change dramatically”. The output of the feature transformer uses a ReLU activation function.
# 
# ![](https://drive.google.com/uc?id=1iuVE-7hkmh2ZMFfY3FdrZ1UbptidK-mI)
# 
# # **<span style="color:#F7B2B0;">Feature Selection :</span>**
# 
# Once features have been transformed, they are passed to the Attentive Transformer and the Mask for feature selection.The Attentive Transformer is comprised of a fully connected layer, batch normalisation and Sparsemax normalisation. It also includes prior scales, meaning it knows how much each feature has been used by the previous steps. This is used to derive the Mask using the processed features from the previous Feature Transformer.
# 
# ![](https://drive.google.com/uc?id=12PNJHZqt7bso16m0H8NZ0wrDq9uLdX0U)
# 
# The Mask ensures the model focuses on the most important features and is also used to derive explainability. It essentially covers up features, meaning the model is only able to use those that have been considered important by the Attentive Transformer.We can also understand feature importance by looking at how much a feature has been masked for all decisions and and an individual prediction.
# TabNet employs soft feature selection with controllable sparsity in end-to-end learning
# This means one model jointly performs feature selection and output mapping, which leads to better performance.TabNet uses instance-wise feature selection, which means features are selected for each input and each prediction can use different features.
# This feature selection is essential as it allows decision boundaries to be generalised to a linear combination of features, where coefficients determine the proportion of each feature, which in the end leads to the model’s interpretability

# In[23]:


X      = df_train[common_features]
y      = df_train["pressure"]
X_test = df_test[common_features]


# In[24]:


X      = X.to_numpy()
y      = y.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()


# In[25]:


regressor = TabNetRegressor(verbose=0,seed=42)
regressor.fit(X_train=X, y_train=y,max_epochs=5,eval_metric=['mae'])


# In[26]:


output = regressor.predict(X_test)


# # **<span style="color:#F7B2B0;">References</span>**
# 
# https://arxiv.org/pdf/1908.07442.pdf
# 
# https://towardsdatascience.com/tabnet-e1b979907694
# 
# @karnikakapoor Header styles 
# 
# @debarshichanda Wandb Content

# # Work in progress 🚧
