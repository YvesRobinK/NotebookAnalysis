#!/usr/bin/env python
# coding: utf-8

# # 1. Imports
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">1.1 Libraries</p>

# In[1]:


# Core
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.6)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
import statistics
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import chi2
from scipy.stats import poisson
import time
import os
from datetime import datetime
import matplotlib.dates as mdates
import plotly.express as px
from termcolor import colored
from sklearn.model_selection import train_test_split, StratifiedKFold
import math
import wandb
from wandb.keras import WandbCallback
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from scipy.special import logit

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.utils import plot_model
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects

rc = {
    "axes.facecolor":"#e5f7e4",
    "figure.facecolor":"#e5f7e4",
    "axes.edgecolor":"#383838"
}

sns.set(rc=rc)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">1.2 Weights & Biases</p>
# 
# W&B is a fantastic **experiment tracking** tool that is completely free. If want to learn how to use it, check out my beginner friendly notebooks:
# 
# * [🐝 Weights & Biases Tutorial (beginner)](https://www.kaggle.com/code/samuelcortinhas/weights-biases-tutorial-beginner)
# * [🐝 Advanced WandB: Hyper-parameter tuning (sweeps)](https://www.kaggle.com/code/samuelcortinhas/advanced-wandb-hyper-parameter-tuning-sweeps)

# In[2]:


from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
wandb_auth = user_secrets.get_secret("wandb_api_key")
get_ipython().system('wandb login $wandb_auth')
os.environ['WANDB_SILENT'] = 'true'


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">1.3 Data</p>

# In[3]:


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s3e2/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s3e2/test.csv', index_col='id')

# Print shape and preview
print('Train shape:', train.shape)
print('Test shape:', test.shape)
train.head()


# **Feature descriptions:** (thanks Sergey)
# 
# * `id`: unique identifier
# * `gender`: categorical = "Male", "Female" or "Other"
# * `age`: float = (0.08 - 82)
# * `hypertension`: bool = (0, 1)
# * `heart_disease`: bool = (0, 1)
# * `ever_married`: categorical = "No", "Yes" (will be converted into bool)
# * `work_type`: categorical = "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# * `Residence_type`: categorical = "Rural" or "Urban"
# * `avg_glucose_level`: float = average glucose level in blood
# * `bmi`: float = body mass index
# * `smoking_status`: categorical = "formerly smoked", "never smoked", "smokes" or "Unknown"*
# * `stroke`: bool = (0, 1)

# # 2. EDA
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">2.1 Missing values</p>
# 
# There are **no missing values**.

# In[4]:


train.isna().sum()


# In[5]:


test.isna().sum()


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">2.2 Data types</p>
# 
# There are **5 categorical** features (object dtype), **3 continuous** features (float64 dtype), 2 integer **binary features** (int64 dtype) and an integer **binary target**.

# In[6]:


train.dtypes


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">2.3 Pairplot</p>
# 
# A **pairplot** shows all the **pairwise relationships** between the features.

# In[7]:


# Pairplot
plt.figure(figsize=(20,20))
plt.suptitle('Pairplot', y=1.05, fontsize=26)
sns.pairplot(data=train, hue='stroke', palette=['#57cf4c', '#378c30'])
plt.show()


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">2.4 Correlations</p>
# 
# A **correlation heatmap** shows us which features share the strongest **linear** relationships. 

# In[8]:


# Heatmap of correlations
plt.figure(figsize=(8,6))
corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(corr, mask=mask,linewidths=.5, cmap='Greens', annot=True)
plt.title('Heatmap of correlations')
plt.show()


# # 3. Pre-processing
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">3.1 Config</p>
# 
# **Configuration dictionary** that we will pass to W&B for **tracking**.

# In[9]:


CFG = dict(
    onehot = True,
    scale = True,
    depth = 6,
    units = 512,
    activation = 'mish',
    dropout_rate = 0.3,
    batch_size = 64,
    epochs = 50,
    verbose = False,
    shuffle = True,
    patience = 10,
    initial_lr = 0.0001,
    n_splits = 10,
    seed = 0,
)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">3.2 Features and labels</p>

# In[10]:


# Labels
y_train = train['stroke']

# Features
X_train = train.drop('stroke', axis=1)
X_test = test


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">3.3 Feature engineering</p>

# In[11]:


# From https://www.kaggle.com/competitions/playground-series-s3e2/discussion/377370
def feature_risk_factors(df):
    df["risk_factors"] = df[[
        "avg_glucose_level", "age", "bmi", 
        "hypertension", "heart_disease", 
        "smoking_status"
    ]].apply(
        lambda x: \
        0 + (1 if x.avg_glucose_level > 99 else 0) + \
        (1 if x.age > 45 else 0) + (1 if x.bmi > 24.99 else 0) + \
        (1 if x.hypertension == 1 else 0) + \
        (1 if x.heart_disease == 1 else 0) + \
        (1 if x.smoking_status in ["formerly smoked", "smokes"] else 0),
        axis=1
    )
    return df


# In[12]:


X_train = feature_risk_factors(X_train)
X_test = feature_risk_factors(X_test)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">3.4 Categorical encoding</p>
# 
# Apply either **one-hot encoding** or **label encoding**. Make the choice a **hyper-parameter**.

# In[13]:


if CFG['onehot']==True:
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
else:
    le = LabelEncoder()
    for col in ['gender','ever_married','work_type','Residence_type','smoking_status']:
        X_train.loc[col] = le.fit_transform(X_train.loc[col])
        X_test.loc[col] = le.transform(X_test.loc[col])


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">3.5 Scale data</p>

# In[14]:


if CFG['scale']:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# # 4. Modelling
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">4.1 Data generator</p>
# 
# The data generator will feed **batches of data** into our model for training and inference. 

# In[15]:


class DataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, shuffle):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Number of steps per epoch'
        return math.ceil(self.X.shape[0] / self.batch_size)
    
    def __getitem__(self, idx):
        'Get minibatch of data'
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_X = self.X[indexes]
        batch_y = self.y[indexes]
        return batch_X, batch_y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">4.2 Helper functions</p>
# 
# Mish is one of the latest developed **activation functions** that reaches state-of-the-art results. 
# 
# * [Mish: A Self Regularized Activation Function [TF]](https://www.kaggle.com/code/samuelcortinhas/mish-a-self-regularized-activation-function-tf)

# In[16]:


# Plot history
def plot_hist(hist):
    history_df = pd.DataFrame(hist.history)
    
    plt.figure(figsize=(22,4))
    plt.subplot(1,3,1)
    plt.plot(history_df['loss'], label='Train_Loss')
    plt.plot(history_df['val_loss'], label='Val_loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(history_df['binary_accuracy'],label='Train_Accuracy')
    plt.plot(history_df['val_binary_accuracy'],label='Val_Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(history_df['auc'],label='Train_AUC')
    plt.plot(history_df['val_auc'],label='Val_AUC')
    plt.title('Area Under Curver')
    plt.legend()
    plt.show()

# Mish using Keras backend
def mish(x):
    '''
    Mish activation function: Mish(x) = x*tanh(softplus(x)) = x*tanh(ln(1+e^x))
    
    Implementated with Keras.
    
    Parameters
    ----------
    x : tf.Tensor. Tensor of shape (n_samples,)
    
    Returns
    -------
    Mish(x) : tf.Tensor. Result of shape (n_samples,)
    
    Example
    -------
    >>> x = tf.convert_to_tensor([1,2,3], dtype=tf.float32)
    >>> z = mish_ke(x)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.8650984, 1.943959 , 2.9865355], dtype=float32)>
    '''
    return x*K.tanh(K.softplus(x))

# Make our custom activation function compatible with keras
get_custom_objects().update({'mish': Activation(mish)})


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">4.3 Neural Network</p>
# 
# Here we **build our model**. Make sure to reference hyper-parameters from the CFG dictionary to make it easier to run experiments and iterate through ideas.

# In[17]:


def build_model():
    # Sequential model
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Hidden layers
    for i in range(CFG['depth']-1):
        model.add(layers.Dense(units=CFG['units']))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(CFG['activation']))
        model.add(layers.Dropout(rate=CFG['dropout_rate']))
    
    # Output layer
    model.add(layers.Dense(units=1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['binary_accuracy','AUC'])
    
    return model


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">4.4 Callbacks</p>
# 
# We use **early stopping** and a **cosine decay learning rate scheduler**. 

# In[18]:


early_stopping = keras.callbacks.EarlyStopping(
    patience = CFG['patience'],
    min_delta = 0.0001,
    monitor = 'val_loss',
    restore_best_weights = True,
)

schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate = CFG['initial_lr'], 
                                                  decay_steps=CFG['epochs'], alpha=0.0001)

scheduler = keras.callbacks.LearningRateScheduler(schedule, verbose=False)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">4.5 Train model</p>

# In[19]:


def train_model():
    # Cross validation
    skf = StratifiedKFold(n_splits=CFG['n_splits'], shuffle=CFG['shuffle'], random_state=CFG['seed'])

    # Initialise
    preds = np.zeros((X_test.shape[0],1))
    val_loss_av = 0
    val_accuracy_av = 0
    val_auc_av = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print('FOLD:', fold)
        # Split data
        y_tr = y_train.iloc[train_idx]
        y_valid = y_train.iloc[val_idx]
        X_tr = X_train.iloc[train_idx,:]
        X_valid = X_train.iloc[val_idx,:]

        # Use generator to shuffle data every epoch
        train_generator = DataGenerator(X_tr.values, y_tr.values, CFG['batch_size'], CFG['shuffle'])

        # Build model
        model = build_model()

        # Train model
        history = model.fit(
            train_generator,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping, scheduler, 
                       #WandbCallback()
                      ],
            batch_size=CFG['batch_size'],
            epochs=CFG['epochs'],
            verbose=CFG['verbose']
        )
        
        # Plot history
        plot_hist(history)

        # Evaluate
        val_preds = model.predict(X_valid)
        val_loss = log_loss(y_valid.astype('float64'), val_preds.astype('float64'))
        val_accuracy = accuracy_score(y_valid, np.round(val_preds))
        val_auc = roc_auc_score(y_valid, val_preds)
        val_loss_av += val_loss/CFG['n_splits']
        val_accuracy_av += val_accuracy/CFG['n_splits']
        val_auc_av += val_auc/CFG['n_splits']
        print('Best validation loss:', val_loss)
        
        # Log metrics
        run.log({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_auc': val_auc})
        
        # Make predictions (soft vote)
        pr = model.predict(X_test)
        preds += pr/CFG['n_splits']
        
    # Log metrics
    run.log({'average_val_loss': val_loss_av, 'average_val_accuracy': val_accuracy_av, 'average_auc': val_auc_av})
    print('\naverage_val_loss', val_loss_av, 'average_val_accuracy', val_accuracy_av, 'average_auc', val_auc_av)
    
    # Save predictions as csv
    sub = pd.read_csv('/kaggle/input/playground-series-s3e2/sample_submission.csv')
    sub['pred'] = preds
    sub.to_csv('submission.csv', index=False)
    
    # Save predictions as Artifact
    artifact = wandb.Artifact(name='submission.csv', type='dataset')
    artifact.add_file('submission.csv')
    run.log_artifact(artifact)
    
    return preds


# In[20]:


# Initialise run
run = wandb.init(entity = 'scortinhas',
                 project = 'ps_s3_e2',
                 config = CFG,
                 save_code = True
)

# Train model
preds = train_model()


# # 5. Submission
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">5.1 Plot predictions</p>

# In[21]:


plt.figure(figsize=(10,4))
sns.histplot(preds, binwidth=0.005, alpha=1)
plt.show()


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">5.2 Save to csv</p>
# 
# Save predictions to a csv file so we can submit to competition.

# In[22]:


# Save predictions as csv
sub = pd.read_csv('/kaggle/input/playground-series-s3e2/sample_submission.csv')
sub['stroke'] = preds
sub.to_csv('submission.csv', index=False)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#207d06; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #207d06;">5.3 Save artifacts</p>
# 
# Save predictions as an artifact in W&B and finish run.

# In[23]:


# Save predictions as Artifact
artifact = wandb.Artifact(name='submission.csv', type='dataset')
artifact.add_file('submission.csv')
run.log_artifact(artifact)


# In[24]:


# Complete run
run.finish()

