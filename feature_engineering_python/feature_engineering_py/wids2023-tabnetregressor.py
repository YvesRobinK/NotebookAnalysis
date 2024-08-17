#!/usr/bin/env python
# coding: utf-8

# # **<span style="color:#F7B2B0;">Goal</span>**
#  
# The WiDS Datathon 2023 focuses on a prediction task involving forecasting sub-seasonal temperatures (temperatures over a two-week period, in this case) within the United States. 
# 
# # **<span style="color:#F7B2B0;">Data</span>**
# 
# The dataset consists of weather and climate information for a number of US locations, for a number of start dates for the two-week observation, as well as the forecasted temperature and precipitation from a number of weather forecast models. Each row in the data corresponds to a single location and a single start date for the two-week period. The task is to predict the arithmetic mean of the maximum and minimum temperature over the next 14 days, for each location and start date.
# 
# # **<span style="color:#F7B2B0;">Evaluation Metric</span>**
# 
# The evaluation metric for this competition is Root Mean Squared Error (RMSE).
# 

# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">
# 
# > I will be integrating W&B for visualizations and logging artifacts!
# > 
# > [WiDS Datathon 2023 Project on W&B Dashboard](https://wandb.ai/usharengaraju/WiDS-Datathon-2023)
# > 
# > - To get the API key, create an account in the [website](https://wandb.ai/site) .
# > - Use secrets to use API Keys more securely 
# 

# In[1]:


pip install pytorch-tabnet


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.preprocessing import LabelEncoder

import optuna
from optuna import Trial, visualization
import torch

import warnings
warnings.filterwarnings('ignore')

from kaggle_secrets import UserSecretsClient
import wandb
from datetime import datetime

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('ggplot')
import seaborn as sns
from scipy import stats

import gc


# In[3]:


# Setup user secrets for login
user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("api_key") 

# Login
wandb.login(key = wandb_api)

CONFIG = dict(competition = 'WiDSDatathon2023',_wandb_kernel = 'tensorgirl')

run = wandb.init(project = "WiDS-Datathon-2023",
                 name = f"Run_{datetime.now().strftime('%d%m%Y%H%M%S')}", 
                 notes = "add some features",
                 tags = [],
                 config = CONFIG,
)


# In[4]:


train=pd.read_csv("/kaggle/input/widsdatathon2023/train_data.csv")
test=pd.read_csv("/kaggle/input/widsdatathon2023/test_data.csv")


# In[5]:


train = train.sample(1000)


# In[6]:


missing_columns = [col for col in train.columns if train[col].isnull().any()]
missingvalues_count =train.isna().sum()
missingValues_df = pd.DataFrame(missingvalues_count.rename('Null Values Count')).loc[missingvalues_count.ne(0)]
missingValues_df .style.background_gradient(cmap="Pastel1")


# In[7]:


# basic stats of features
train.describe().style.background_gradient(cmap="Pastel1")


# In[8]:


plt.figure(figsize=(15, 7))
plt.subplot(121)
sns.kdeplot(train['contest-tmp2m-14d__tmp2m'] , color = "#ffd514")
plt.subplot(122)
sns.boxplot(train['contest-tmp2m-14d__tmp2m'] , color = "#ff355d")


# In[9]:


res = stats.probplot(train['contest-tmp2m-14d__tmp2m'], plot=plt)


# In[10]:


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


# In[11]:


numerical_features = ['nmme0-tmp2m-34w__nasa0', 'nmme-tmp2m-56w__cancm3', 'nmme-tmp2m-56w__gfdlflorb','nmme-prate-34w__cancm4','nmme-prate-34w__gfdlflora']


# # **<span style="color:#F7B2B0;">Distribution of Numeric Variables</span>**

# In[12]:


# plot distributions of features
for feature in numerical_features:
    kdeplot_features(train,test, feature=feature, title = feature + " distribution")


# # **<span style="color:#F7B2B0;">Distribution of Categorical Variables</span>**

# In[13]:


countplot_features(train, feature='climateregions__climateregion', title = "Frequency of "+ feature)


# # **<span style="color:#F7B2B0;"> Log Plots to W&B environment </span>**

# In[14]:


# Log Plots to W&B environment
title = "Distribution of Numerical features"
run = wandb.init(project='WiDSDatathon2023', name=title,config=CONFIG)
for feature in numerical_features:
    title = "Distribution of Numerical "+feature    
    create_wandb_hist(x_data=train[feature],x_name=feature , title=title,log="hist")    
wandb.finish()

title = "Countplot Distribution"
run = wandb.init(project='WiDSDatathon2023', name=title,config=CONFIG) 
table = wandb.Table(data=train, columns=['climateregions__climateregion'])
wandb.log({'categorical feature': wandb.plot.histogram(table, 'climateregions__climateregion', title=title)})


# # **<span style="color:#F7B2B0;">Feature Engineering</span>**

# In[15]:


train['year']=pd.DatetimeIndex(train['startdate']).year 
train['month']=pd.DatetimeIndex(train['startdate']).month 
train['day']=pd.DatetimeIndex(train['startdate']).day

test['year']=pd.DatetimeIndex(test['startdate']).year 
test['month']=pd.DatetimeIndex(test['startdate']).month 
test['day']=pd.DatetimeIndex(test['startdate']).day


# # **<span style="color:#F7B2B0;">Label Encoding</span>**

# In[16]:


le = LabelEncoder()
train['climateregions__climateregion'] = le.fit_transform(train['climateregions__climateregion'])
test['climateregions__climateregion'] = le.transform(test['climateregions__climateregion'])


# In[17]:


## remove the irrelevant columns
train=train.drop(['index'],axis=1)
train=train.drop(['startdate'],axis=1)


test=test.drop(['index'],axis=1)
test=test.drop(['startdate'],axis=1)


# # **<span style="color:#F7B2B0;">Missing Values Imputation</span>**

# In[18]:


train = train.dropna()


# # **<span style="color:#F7B2B0;"> Feature Correlation </span>**

# In[19]:


from yellowbrick.target import FeatureCorrelation

X1 = train.drop(columns=['contest-tmp2m-14d__tmp2m'])
y1 = train['contest-tmp2m-14d__tmp2m']

# Create a list of the feature names
features = np.array(X1.columns)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

visualizer.fit(X1, y1)        
visualizer.show() 


# In[ ]:





# # **<span style="color:#F7B2B0;">W & B Artifacts</span>**
# 
# An artifact as a versioned folder of data.Entire datasets can be directly stored as artifacts .
# 
# W&B Artifacts are used for dataset versioning, model versioning . They are also used for tracking dependencies and results across machine learning pipelines.Artifact references can be used to point to data in other systems like S3, GCP, or your own system.
# 
# You can learn more about W&B artifacts [here](https://docs.wandb.ai/guides/artifacts)
# 
# ![](https://drive.google.com/uc?id=1JYSaIMXuEVBheP15xxuaex-32yzxgglV)

# In[20]:


# Save train data to W&B Artifacts
train.to_csv("train_features.csv", index = False)
run = wandb.init(project='WiDSDatathon2023', name='training_data',config=CONFIG) 
artifact = wandb.Artifact(name='training_data',type='dataset')
artifact.add_file("./train_features.csv")

wandb.log_artifact(artifact)
wandb.finish()


# In[21]:


from sklearn.model_selection import KFold
X = train.drop(columns=['contest-tmp2m-14d__tmp2m']).values
y = train['contest-tmp2m-14d__tmp2m'].values
y = y.reshape(-1, 1)


# # **<span style="color:#F7B2B0;">TabNet</span>**
# 
# [Source](https://towardsdatascience.com/tabnet-e1b979907694)
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

# In[22]:


def Objective(trial):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_da = trial.suggest_int("n_da", 56, 64, step=4)
    n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
    gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 3)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                     lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type=mask_type, n_shared=n_shared,
                     scheduler_params=dict(mode="min",
                                           patience=trial.suggest_int("patienceScheduler",low=3,high=10), # changing sheduler patience to be lower than early stopping patience 
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     ) #early stopping
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    CV_score_array    =[]
    for train_index, test_index in kf.split(X):
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        regressor = TabNetRegressor(**tabnet_params)
        regressor.fit(X_train=X_train, y_train=y_train,
                  eval_set=[(X_valid, y_valid)],
                  patience=trial.suggest_int("patience",low=15,high=30), max_epochs=trial.suggest_int('epochs', 1, 100),
                  eval_metric=['rmse'])
        CV_score_array.append(regressor.best_cost)
    avg = np.mean(CV_score_array)
    return avg


# # **<span style="color:#F7B2B0;">Hyperparameter Tuning Using Optuna</span>**
# 
# [Source](https://github.com/optuna/optuna)
# 
# Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. The code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.
# 
# **Key Features**
# 
# Optuna has modern functionalities as follows:
# 
# 📌 **Lightweight, versatile, and platform agnostic architecture**
# 
# Handle a wide variety of tasks with a simple installation that has few requirements.
# 
# 📌 **Pythonic search spaces**
# 
# Define search spaces using familiar Python syntax including conditionals and loops.
# 
# 📌 **Efficient optimization algorithms**
# 
# Adopt state-of-the-art algorithms for sampling hyperparameters and efficiently pruning unpromising trials.
# 
# 📌 **Easy parallelization**
# 
# Scale studies to tens or hundreds or workers with little or no changes to the code.
# 
# 📌 **Quick visualization**
# 
# Inspect optimization histories from a variety of plotting functions.

# In[23]:


study = optuna.create_study(direction="minimize", study_name='TabNet optimization')
study.optimize(Objective, timeout=6*60) #5 hours


# In[24]:


print(study.best_params)
TabNet_params = study.best_params


# In[25]:


final_params = dict(n_d=TabNet_params['n_da'], n_a=TabNet_params['n_da'], n_steps=TabNet_params['n_steps'], gamma=TabNet_params['gamma'],
                     lambda_sparse=TabNet_params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type=TabNet_params['mask_type'], n_shared=TabNet_params['n_shared'],
                     scheduler_params=dict(mode="min",
                                           patience=TabNet_params['patienceScheduler'],
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     )
epochs = TabNet_params['epochs']


# In[26]:


regressor = TabNetRegressor(**final_params)
regressor.fit(X_train=X, y_train=y,
          patience=TabNet_params['patience'], max_epochs=epochs,
          eval_metric=['rmse'])


# # **<span style="color:#F7B2B0;">Making Submission</span>**

# In[27]:


submission = pd.read_csv('/kaggle/input/widsdatathon2023/sample_solution.csv')
submission['contest-tmp2m-14d__tmp2m'] = regressor.predict(test.values)
submission.to_csv('submission',index = False)


# In[28]:


display(submission)


# # **<span style="color:#F7B2B0;">References</span>**
# 
# 
# https://github.com/optuna/optuna
# 
# https://towardsdatascience.com/tabnet-e1b979907694
# 
