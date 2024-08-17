#!/usr/bin/env python
# coding: utf-8

# This is my journey to tackle this Kaggle competition. Every version, I will try to update my work to improve my accuracy. Let me know how I am doing and if you have any suggestions!
# 
# * Version 2
#     * Run a Baseline model
# * Version 5
#     * implementing feature engineering
#         * removed columns: 'Soil_Type7' and 'Soil_Type15'
#         * clipped hillshade variables to fit in range [0-255]
#         * tweaked 'Aspect' to have a range from 0-360 degrees
#         * Resources: https://www.kaggle.com/ambrosm/tpsdec21-01-keras-quickstart
#     * train-valid split
#         * changed train-valid split from 70/30 to 80/20
#         * added straify parameter in train_test_split
#     * model
#         * added two more layers
#         * changed activation function of hidden layers to ReLU
# * Version 6
#     * scale the features
#     * un-dummify the 'Soil-Type's and apply frequency encoding
# * Version 7
#     * un-dummify the 'Wilderness-Type's and apply frequency encoding
#     * changed batch size from 2048 -> 1024
#     * removed unnecessary comments/ code and added more documentation

# In[1]:


# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow as tf


# In[2]:


# Reading in the data

train = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')


# In[3]:


# Function to reduce memory

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[4]:


# Reducing memory usage

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[5]:


# Viewing the summary statistics of the data

# train.describe().T


# In[6]:


# Viewing the info of the 

# train.info()


# In[7]:


# Viewing the number of categories in target variable

train['Cover_Type'].nunique()


# In[8]:


# Viewing the number of observations per category in target variable

train['Cover_Type'].value_counts()


# ## Feature Engineering

# ### Soil Variables

# It will be a good idea to remove 'Soil_Type7' and 'Soil_Type15'because it is 0's for all observations. Therefore it is not informative and might add noise to the model.

# In[9]:


# Remove columns 'Soil_Type7', 'Soil_Type15'

train.drop(['Soil_Type7', 'Soil_Type15'], inplace=True, axis=1)
test.drop(['Soil_Type7', 'Soil_Type15'], inplace=True, axis=1)


# In[10]:


# Extracting soil columns

soil_columns = [col for col in train.columns if 'Soil' in col]


# In[11]:


# Undummying the Soil_types

train['soil_type'] = train[soil_columns].idxmax(axis=1)
test['soil_type'] = test[soil_columns].idxmax(axis=1)


# In[12]:


# Calculating the fequency encoding

soil_map = pd.Series(train['soil_type'].value_counts()/train.shape[0]).to_dict()


# In[13]:


# Applying the frequency encoding

train['soil_type'] = train['soil_type'].map(soil_map)
test['soil_type'] = test['soil_type'].map(soil_map)


# In[14]:


# Dropping all the 'Soil-Type' columns

train = train.drop(soil_columns, axis=1)
test = test.drop(soil_columns, axis=1)
train.head()


# ### Wilderness Variables

# In[15]:


# Extracting winderness columns

wild_columns = [col for col in train.columns if 'Wild' in col]


# In[16]:


# Undummying the wilderness_types

train['wild_type'] = train[wild_columns].idxmax(axis=1)
test['wild_type'] = test[wild_columns].idxmax(axis=1)


# In[17]:


# Calculating the fequency encoding

wild_map = pd.Series(train['wild_type'].value_counts()/train.shape[0]).to_dict()


# In[18]:


# Applying the frequency encoding

train['wild_type'] = train['wild_type'].map(wild_map)
test['wild_type'] = test['wild_type'].map(wild_map)


# In[19]:


# Dropping all the 'Soil-Type' columns

train = train.drop(wild_columns, axis=1)
test = test.drop(wild_columns, axis=1)
train.head()


# Hillshade is an "image" that ranges from 0-255. However some of the hillshade values are less than 0 or greater than 255. We will make an assumption that those were data entry errors and will clip them. If it is less than 0, we will set it to sero. If it is greater than 255, set it to 255.
# 
# Some additional thoughts:
# * Set values under 0 to 0 and values greater than 255 to 255 for all Hillshade variables
# * Is clipping the best way to procede? Try just scaling instead of clipping
# * Remove the hillshade data that are NOT within the range between 0, 255

# In[20]:


# Clipping the hillshade columns between 0 and 255

hillshade_columns = [col for col in train.columns if 'Hillshade' in col]

for col in hillshade_columns:
    train[col] = train[col].clip(0,255)
    test[col] = test[col].clip(0,255)


# In[21]:


# Changing the range of Aspect to fall between 0 and 359

train['Aspect'] = train['Aspect'].apply(lambda row: row%360)
test['Aspect'] = test['Aspect'].apply(lambda row: row%360)
train['Aspect'].describe()


# In[22]:


# Getting the features and target variables

features = [col for col in train.columns if col not in ['Id', 'Cover_Type']]
target = 'Cover_Type'


# In[23]:


# Label encoding the target variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train[target] = le.fit_transform(train[target])


# In[24]:


# Removing that single observation that has tree type '5' (or '4' after LabelEncoding)

train = train.loc[train['Cover_Type'] != 4,]


# In[25]:


# # Saving preprocessed data

# train.to_csv('train_reduced.csv', index=False)
# test.to_csv('test_reduced.csv', index=False)


# In[26]:


# Viewing the shape of the features

train[features].shape


# In[27]:


# Splitting the data into train and test splot

X_train, X_valid, y_train, y_valid = train_test_split(
    train[features], 
    train[target],
    stratify=train[target],
    test_size=0.2, 
    random_state=0
)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of X_valid: {X_valid.shape}')
print(f'Shape of y_valid: {y_valid.shape}')


# ## Scaling the data

# In[28]:


# Scaling the data by fitting on X_train and scaling the rest

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
t = scaler.transform(test[features])


# ### Tensorflow Model

# In[29]:


# Function that creates a TF sequential model

def get_model():
    tf.keras.backend.clear_session()

    ## Creating a Sequential Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_shape=(None,12), activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(7, activation = 'softmax')
    ])
    
    ## Compile 
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=['acc']
    )
    
    return model


# In[30]:


# K-fold Cross Validation model evaluation

from sklearn.metrics import accuracy_score

X = X_train
y = y_train.values

FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 1024

test_preds = np.zeros((1, 1))
scores = []

cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_t, X_v = X[train_idx], X[val_idx]
    y_t, y_v = y[train_idx], y[val_idx]
    
        # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold} ...')
    
    model = get_model()

    # Fit data to model
    model.fit(
        X_t,
        y_t,
        validation_data=(X_v, y_v),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    
    y_pred = np.argmax(model.predict(X_v), axis=1)
    score = accuracy_score(y_v, y_pred)
    scores.append(score)


# In[31]:


# Printing the results from K-Fold

print(f'Accuracy for each fold: {scores}')
print(f'Mean of all the folds: {np.mean(scores):.4f}')
print(f'Standard Deviation of the folds: {np.std(scores):.4f}')


# Compared to Version 6, model has not imporved much. Makes me wonder if the wilderness columns are significant to predict the tree type. Some next steps can be to improve the NN architecture. And possibly apply blending.

# In[32]:


# Predicting on the test set

preds = model.predict(t)


# In[33]:


# Reversing the label encoder

final_preds = le.inverse_transform(preds.argmax(axis=1))


# In[34]:


# Creating a submission file

submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': final_preds })
submission.to_csv('submission.csv', index=False)


# In[ ]:




