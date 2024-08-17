#!/usr/bin/env python
# coding: utf-8

# 
# [Source](https://jmlr.csail.mit.edu/papers/volume22/20-1223/20-1223.pdf)
# 
# [Github](https://github.com/suinleelab/path_explain)
# 
# 
# Recent work has shown great promise in explaining neural network behavior. In particular, feature attribution methods explain which features were most important to a model’s prediction on a given input. However, for many tasks, simply knowing which features were important to a model’s prediction may not provide enough insight to understand model behavior. The interactions between features within the model may better help us understand not only the model, but also why certain features are more important than others.
# 
# In the paper ”Explaining Explanations: Axiomatic Feature Interactions for Deep Networks” the authors present Integrated Hessians2 : an extension of Integrated Gradients that explains pairwise feature interactions in neural networks. Integrated Hessians overcomes several theoretical limitations of previous methods to explain interactions, and unlike such previous methods is not limited to a specific architecture or class of neural network. Additionally, the proposed method is faster than existing methods when the number of features is large, and outperforms previous methods on existing quantitative benchmarks.
# 
# # **<span style="color:#F7B2B0;">Feature attribution: </span>**
# 
# There have been a large number of recent approaches to interpret deep neural networks, ranging from methods that aim to distill complex models into more simple models to methods that aim to identify the most important concepts learned by a network. One of the best-studied sets of approaches is known as feature attribution methods and these approaches explain a model’s prediction by assigning credit to each input feature based on how much it influenced that prediction. Although these approaches help practitioners understand which features are important, they do not explain why certain features are important or how features interact in a model. In order to develop a richer understanding of model behavior, it is therefore desirable to develop methods to explain not only feature attributions but also feature interactions. 
# 
# # **<span style="color:#F7B2B0;">Feature interaction: </span>**
# 
# 
# There are several existing methods that explain feature interactions in neural networks. One of the papers proposes a method to explain global interactions in Bayesian Neural Networks (BNN) by examining pairs of features that have large second-order derivatives at the input. Neural Interaction Detection is a method that detects statistical interactions between features by examining the weight matrices of feed-forward neural networks .
# 
# # **<span style="color:#F7B2B0;">Limitations of Prior Approaches: </span>**
# 
# While previous approaches have taken important steps towards understanding feature interaction in neural networks, all suffer from practical limitations, including being limited to specific types of architectures. Neural Interaction Detection only applies to feedforward neural network architectures, and can not be used on networks with convolutions, recurrent units, or self-attention. Contextual Decomposition has been applied to LSTMs, feed-forward neural networks and convolutional networks, but to our knowledge is not straightforward to apply to more recent innovations in deep learning, such as self-attention layers. 
# 
# # **<span style="color:#F7B2B0;">Explaining Explanations: Axiomatic Feature Interactions for Deep Networks: </span>**
# 
# The paper proposes an approach to quantify pairwise feature interactions that can be applied to any neural network architecture .It identifies several common-sense axioms that feature-level interactions should satisfy and show that the proposed method satisfies them.The paper provides a principled way to compute interactions in ReLU-based networks, which are piece-wise linear and have zero second derivatives.The paper evaluates our method against existing methods and shows that it better identifies interactions in simulated data.
# 
# [Source](https://jmlr.csail.mit.edu/papers/volume22/20-1223/20-1223.pdf)
# 
# [Github](https://github.com/suinleelab/path_explain)
# 

# <img src="https://camo.githubusercontent.com/dd842f7b0be57140e68b2ab9cb007992acd131c48284eaf6b1aca758bfea358b/68747470733a2f2f692e696d6775722e636f6d2f52557469567a482e706e67">
# 
# > I will be integrating W&B for visualizations and logging artifacts!
# > 
# > [Playground Series Project on W&B Dashboard]https://wandb.ai/usharengaraju/PBSEpisode3)
# > 
# > - To get the API key, create an account in the [website](https://wandb.ai/site) .
# > - Use secrets to use API Keys more securely 

# In[1]:


get_ipython().system('pip install path-explain')


# In[2]:


import tensorflow as tf

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

from path_explain import PathExplainerTF
from path_explain.utils import set_up_environment
#from plot.scatter import scatter_plot, _set_axis_config
#from plot.summary import summary_plot

from path_explain import PathExplainerTF, scatter_plot, summary_plot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.layers import Input, Dense, Dropout
from keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam

import wandb
from kaggle_secrets import UserSecretsClient
from datetime import datetime

from wandb.keras import WandbCallback


# In[3]:


# Setup user secrets for login
user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("api_key") 

# Login
wandb.login(key = wandb_api)

run = wandb.init(project = "PBSEpisode3",
                 name = f"Run_{datetime.now().strftime('%d%m%Y%H%M%S')}", 
                 notes = "add some features",
                 tags = [],
                 config = dict(competition = 'PBSEpisode3',
                               _wandb_kernel = 'tensorgirl',
                               batch_size = 32,
                               epochs = 30,
                               learning_rate = 0.005)
)

config = wandb.config


# In[4]:


train = pd.read_csv('../input/playground-series-s3e3/train.csv',index_col=[0])
test = pd.read_csv('../input/playground-series-s3e3/test.csv',index_col=[0])
submission = pd.read_csv('../input/playground-series-s3e3/sample_submission.csv')


# In[5]:


train.columns


# # **<span style="color:#F7B2B0;">Target Value Distribution </span>**

# In[6]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
(train.Attrition.value_counts(normalize=True)*100).plot(kind='bar', color=['mediumseagreen', 'lightcoral'])
display(train.Attrition.value_counts(normalize=True)*100)
ax.set_ylim([0, 100])
ax.set_ylabel('Attrition of Employees [%]', fontsize=14)
ax.set_xticklabels(['0', '1'], fontsize=14, rotation=0)
plt.show()


# In[7]:


categorical_columns = train.select_dtypes(include="object").columns

fig, ax = plt.subplots(7,1, figsize=(20, 30))
bins = 30
for feature, ax in zip(categorical_columns, ax.flatten()):
    sns.histplot(train[feature], bins=bins, ax=ax, color='lightcoral', alpha=0.7, label='Train' ,edgecolor='black')
    sns.histplot(test[feature], bins=bins, ax=ax, alpha=0.9, color='mediumseagreen', edgecolor='black', label='Test')
    # Add a legend and labels
    ax.legend(loc='upper right', fontsize=16)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=12)

# Tighten the layout and show the figure
plt.tight_layout()
plt.show()


# In[8]:


numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'EmployeeCount', 'HourlyRate',
       'JobLevel', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

fig, ax = plt.subplots(18, 1, figsize=(20, 60))
bins = 30

for feature, ax in zip(numerical_features, ax.flatten()):
    sns.histplot(train[feature], bins=bins, ax=ax, color='lightcoral', alpha=0.7, label='Train' ,edgecolor='black')
    sns.histplot(test[feature], bins=bins, ax=ax, alpha=0.9, color='mediumseagreen', edgecolor='black', label='Test')
    # Add a legend and labels
    ax.legend(loc='upper right', fontsize=16)
    ax.set_xlabel(feature, fontsize=16)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=16)

# Tighten the layout and show the figure
plt.tight_layout()
plt.show()


# # **<span style="color:#F7B2B0;">Feature Engineering </span>**

# In[9]:


# Drop Over18 column since it contains the same value
train.drop("Over18", axis=1, inplace=True)
test.drop("Over18", axis=1, inplace=True)


# In[10]:


# code copied from https://www.kaggle.com/code/oscarm524/ps-s3-ep3-eda-modeling

## Replacing weird Education value
train.loc[train['Education'] == 15, 'Education'] = 5
test.loc[test['Education'] == 15, 'Education'] = 5

train_FE = train.copy()
test_FE = test.copy()

## Defining inputs and target
train_dummies = pd.get_dummies(train_FE[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']])
train_dummies = train_dummies.drop(columns = ['BusinessTravel_Non-Travel', 'Department_Research & Development', 'EducationField_Other', 'Gender_Female', 'JobRole_Manufacturing Director', 'MaritalStatus_Divorced', 'OverTime_No'])

test_dummies = pd.get_dummies(test_FE[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']])
test_dummies = test_dummies.drop(columns = ['BusinessTravel_Non-Travel', 'Department_Research & Development', 'EducationField_Other', 'Gender_Female', 'JobRole_Manufacturing Director', 'MaritalStatus_Divorced', 'OverTime_No'])


X = train_FE.drop(columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'StandardHours', 'Attrition'], axis = 1)
X = pd.concat([X, train_dummies], axis = 1)
y = train_FE['Attrition']

X_test = test_FE.drop(columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'StandardHours'], axis = 1)
X_test = pd.concat([X_test, test_dummies], axis = 1)

X_wandb = pd.concat([X, y], axis = 1)


# In[11]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# In[12]:


sc = StandardScaler()
X_train1 = sc.fit_transform(X_train)
X_val1  = sc.transform(X_val)
X_test1  = sc.transform(X_test)

X_train1 = X_train1.astype(np.float32)
X_val1 = X_val1.astype(np.float32)
X_test1  = X_test1.astype(np.float32)


# # **<span style="color:#F7B2B0;">W & B Artifacts</span>**
# 
# An artifact as a versioned folder of data.Entire datasets can be directly stored as artifacts .
# 
# W&B Artifacts are used for dataset versioning, model versioning . They are also used for tracking dependencies and results across machine learning pipelines.Artifact references can be used to point to data in other systems like S3, GCP, or your own system.
# 
# You can learn more about W&B artifacts [here](https://docs.wandb.ai/guides/artifacts)
# 
# ![](https://drive.google.com/uc?id=1JYSaIMXuEVBheP15xxuaex-32yzxgglV)

# In[13]:


# Save train data to W&B Artifacts
X_wandb.to_csv("train_features.csv", index = False)
run = wandb.init(project='PBSEpisode3', name='training_data',config=config) 
artifact = wandb.Artifact(name='training_data',type='dataset')
artifact.add_file("./train_features.csv")

wandb.log_artifact(artifact)
wandb.finish()


# In[14]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(45,)))
model.add(Dense(64, activation=relu))
model.add(Dropout(0.2))
model.add(Dense(32, activation=relu))
model.add(Dropout(0.1))
model.add(tf.keras.layers.Dense(units=1,  activation=sigmoid))


# Define the optimizer
optimizer = Adam(learning_rate=config.learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
    
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.keras.metrics.AUC()]
model.compile(optimizer=optimizer,
          loss=loss,
          metrics=metrics)

run = wandb.init()

history = model.fit(X_train1, y_train, batch_size=config.batch_size,epochs=config.epochs, verbose=0 ,validation_split = 0.2,callbacks = [WandbCallback()])


# In[15]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 1])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


# In[16]:


plt.plot(history.history['auc'], label='auc')
plt.plot(history.history['val_auc'], label='val_auc')
plt.ylim([0, 1])
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()


# In[17]:


from IPython import display

# we create an IFrame and set the width and height
iF = display.IFrame(run.url, width=1080, height=720)
iF


# In[18]:


train_loss, train_auc = model.evaluate(X_train1, y_train, batch_size=51, verbose=0)
val_loss , val_auc = model.evaluate(X_val1, y_val, batch_size=51, verbose=0)


# In[19]:


print('Train loss: {:.4f}\tTrain AUC: {:.4f}'.format(train_loss, train_auc))
print('Test loss: {:.4f}\tValidation AUC: {:.4f}'.format(val_loss, val_auc))


# In[20]:


y_pred = model.predict(X_test1)
y_pred_discrete = (y_pred > 0.5).astype(int)[:, 0]
submission['Attrition'] = y_pred_discrete
submission.to_csv("submission.csv",index=False)
submission.head()


# In[21]:


explainer = PathExplainerTF(model)


# In[22]:


all_data = np.concatenate([X_train1, X_val1], axis=0)


# In[23]:


attributions = explainer.attributions(inputs=all_data,
                                      baseline=X_train1,
                                      batch_size=32,
                                      num_samples=200,
                                      use_expectation=True,
                                      output_indices=0,
                                      verbose=True)


# In[24]:


interactions = explainer.interactions(inputs=all_data,
                                      baseline=X_train1,
                                      batch_size=100,
                                      num_samples=200,
                                      use_expectation=True,
                                      output_indices=0,
                                      verbose=True)


# In[25]:


all_data_renorm = np.concatenate([X_train, X_val])


# In[26]:


summary_plot(attributions,
             all_data_renorm,
             interactions=None,
             interaction_feature=None,
             feature_names=X_train.columns,
             plot_top_k=10)


# ## References
# 
# https://www.kaggle.com/code/oscarm524/ps-s3-ep3-eda-modeling
# 
# https://www.kaggle.com/code/maazkarim/eda-fe-tf-keras
# 
# https://jmlr.csail.mit.edu/papers/volume22/20-1223/20-1223.pdf
# 
# https://github.com/suinleelab/path_explain
