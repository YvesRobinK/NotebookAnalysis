#!/usr/bin/env python
# coding: utf-8

# # Acknowledgement
# ### Thanks to [@cdeotte](https://www.kaggle.com/cdeotte) for sharing [this](https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153) great finding. Surely helped increase my score.

# # GPU Info

# In[1]:


get_ipython().system('nvidia-smi')


# # Imports

# In[2]:


# Asthetics
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from IPython.display import display
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import glob
import random
import gc
gc.enable()
pd.set_option('display.max_columns', None)

# Utils
from sklearn import preprocessing
# Deep Learning
import tensorflow as tf
from tensorflow import keras
# Metrics
from sklearn.metrics import mean_absolute_error

# Random Seed Initialize
RANDOM_SEED = 42

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything()


# In[3]:


data_dir = '../input/ventilator-pressure-prediction'
train_file_path = os.path.join(data_dir, 'train.csv')
test_file_path = os.path.join(data_dir, 'test.csv')
sample_sub_file_path = os.path.join(data_dir, 'sample_submission.csv')

models_dir = '../input/google-brain-ventilator-tf-lstm-models/50_Features'

print(f'Train file: {train_file_path}')
print(f'Test file: {test_file_path}')
print(f'Sample Sub file: {sample_sub_file_path}')


# In[4]:


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
sub_df = pd.read_csv(sample_sub_file_path)


# In[5]:


display(test_df.head())
print(test_df.shape)


# In[6]:


all_pressure = np.sort(train_df.pressure.unique())
PRESSURE_MIN = all_pressure[0].item()
PRESSURE_MAX = all_pressure[-1].item()
PRESSURE_STEP = ( all_pressure[1] - all_pressure[0] ).item()


# # Feature Engineering

# In[7]:


# From https://www.kaggle.com/tenffe/finetune-of-tensorflow-bidirectional-lstm
def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    return df


# In[8]:


train_df = add_features(train_df)
test_df = add_features(test_df)

display(test_df.head())
print(test_df.shape)


# In[9]:


train_df.drop(['id', 'pressure', 'breath_id'], axis=1, inplace=True)
test_df = test_df.drop(['id', 'breath_id'], axis=1)

scaler = preprocessing.RobustScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

del train_df
gc.collect()

X_test = test_df.reshape(-1, 80, test_df.shape[-1])


# # Prediction

# In[10]:


predicted_labels = []
for model_name in glob.glob(models_dir + '/*.h5'):
    print(model_name)
    model = tf.keras.models.load_model(model_name)
    predictions = model.predict(X_test).squeeze().reshape(-1, 1).squeeze()
    predicted_labels.append(predictions)
    
    del model
    gc.collect()


# In[11]:


sub_df['pressure'] = sum(predicted_labels)/(len(glob.glob(models_dir + '/*.h5')))
display(sub_df.head())
sub_df.to_csv('submission_mean.csv', index=False)


# In[16]:


# From https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153
sub_df['pressure'] = np.median(np.vstack(predicted_labels),axis=0)*.75+np.mean(np.vstack(predicted_labels),axis=0)*.25
display(sub_df.head())
sub_df.to_csv('submission_median.csv', index=False)


# In[17]:


# From https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153
sub_df['pressure'] = np.round((sub_df.pressure - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN
sub_df.pressure = np.clip(sub_df.pressure, PRESSURE_MIN, PRESSURE_MAX)
display(sub_df.head())
sub_df.to_csv('submission.csv', index=False)


# In[ ]:




