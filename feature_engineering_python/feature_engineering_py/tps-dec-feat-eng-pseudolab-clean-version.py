#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The objective to predict the cover type of a forest given features like elavation, soil type etc. There are 7 different cover types to predict in total.
# 
# ![https://th.bing.com/th/id/OIP.PcAN1kc44gDpHowTie715gHaD4?pid=ImgDet&rs=1](https://th.bing.com/th/id/OIP.PcAN1kc44gDpHowTie715gHaD4?pid=ImgDet&rs=1)

# **Acknowledgments:**
# * [Confusion matrices](https://www.kaggle.com/ambrosm/tpsdec21-01-keras-quickstart) by [AmbrosM](https://www.kaggle.com/ambrosm).
# * [Feature engineering](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293373) by [Gulshan Mishra](https://www.kaggle.com/gulshanmishra).
# * [Memory usage](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/291844) by [Luca Massaron](https://www.kaggle.com/lucamassaron).
# * [Ensembling](https://www.kaggle.com/odins0n/tps-dec-eda-modeling/notebook#Modeling) by [Sanskar Hasija
# ](https://www.kaggle.com/odins0n).
# * [Pseudolabelling](https://www.kaggle.com/remekkinas/tps-12-nn-tpu-pseudolabeling-0-95661/notebook) by [Remek Kinas](https://www.kaggle.com/remekkinas).

# This notebook will be essentially the same to my other notebook, except we won't do EDA here to save memory. This will allow use to use more folds in the cross validation stage. 
# 
# See below for my main notebook:
# * [EDA, Feature Engineering and Pseudolabelling](https://www.kaggle.com/samuelcortinhas/tps-dec-eda-feat-eng-pseudolab)

# # Libraries

# In[1]:


# Core
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.get_option("display.max_columns", 55)
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
import statistics
import time

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


# # Data

# In[2]:


# Save to df
train_data=pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv', index_col='Id')
test_data=pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv', index_col='Id')

# save for submission
test_index=test_data.index

# Shape and preview
print('Training data df shape:',train_data.shape)
print('Test data df shape:',test_data.shape)
train_data.head()


# # Pseudolabeling

# In[3]:


# Save to df
pseudo_label_df=pd.read_csv('../input/tps12-pseudolabels/tps12-pseudolabels_v2.csv', index_col='Id')

# Concatenate
new_train_data=pd.concat([train_data, pseudo_label_df], axis=0)

# Remove pseudolabel samples from test set
pseudo_label_index=pseudo_label_df.index
new_test_data=test_data.drop(pseudo_label_index, axis=0)

# Save for submission
new_test_data_index=new_test_data.index
pseudo_label_preds_df=pd.DataFrame({'Id': pseudo_label_index,
                       'Cover_Type': pseudo_label_df['Cover_Type']}).reset_index(drop=True)


# **Drop label 5**

# In[4]:


new_train_data.drop(new_train_data[new_train_data.Cover_Type==5].index, axis=0, inplace=True)


# # Feature engineering

# **Remove unwanted negative values**

# For the features below it does not make physical sense to include negative numbers.

# In[5]:


# Specify features to clip
mask_features=['Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
              'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']

# Clip negative values
new_train_data[mask_features]=new_train_data[mask_features].clip(lower=0)
new_test_data[mask_features]=new_test_data[mask_features].clip(lower=0)


# **Aspect**

# Aspect values represent angles between 0 and 360 degrees so we should project them onto [0,360] to make any patterns easier to learn.

# In[6]:


# Project training aspect angles onto [0,360]
new_train_data['Aspect'][new_train_data['Aspect'] < 0] += 360
new_train_data['Aspect'][new_train_data['Aspect'] >= 360] -= 360

# Project test aspect angles onto [0,360]
new_test_data['Aspect'][new_test_data['Aspect'] < 0] += 360
new_test_data['Aspect'][new_test_data['Aspect'] >= 360] -= 360


# **Distance to Hydrology**

# We have the horizontal and vertical distances to Hydrology so we can use these to calculate the l1 or euclidean distance.

# In[7]:


# l1 (aka Manhattan) distance to Hydrology
new_train_data['l1_Hydrology'] = np.abs(new_train_data['Horizontal_Distance_To_Hydrology']) + np.abs(new_train_data['Vertical_Distance_To_Hydrology'])
new_test_data['l1_Hydrology'] = np.abs(new_test_data['Horizontal_Distance_To_Hydrology']) + np.abs(new_test_data['Vertical_Distance_To_Hydrology'])


# In[8]:


# Euclidean distance to Hydrology (training set)
new_train_data["ED_to_Hydrology"] = np.sqrt((new_train_data['Horizontal_Distance_To_Hydrology'].astype(np.int32))**2 + 
                                        (new_train_data['Vertical_Distance_To_Hydrology'].astype(np.int32))**2)

# Euclidean distance to Hydrology (test set)
new_test_data["ED_to_Hydrology"] = np.sqrt((new_test_data['Horizontal_Distance_To_Hydrology'].astype(np.int32))**2 + 
                                       (new_test_data['Vertical_Distance_To_Hydrology'].astype(np.int32))**2)


# **Hillshade**

# From [ArcMap](https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/hillshade-function.htm): "A hillshade is a grayscale 3D representation of the surface, with the sun's relative position taken into account for shading the image." 
# 
# This means all Hillshade values should lie in the range [0, 255] because it corresponds to a greyscale image.

# In[9]:


# Clips hillshades 0 to 255 index
hillshades = [col for col in train_data.columns if col.startswith('Hillshade')]

# Clip df's
new_train_data[hillshades] = new_train_data[hillshades].clip(0, 255)
new_test_data[hillshades] = new_test_data[hillshades].clip(0, 255)


# **Number of soil & wilderness types**

# Credit: [Craig Thomas](https://www.kaggle.com/craigmthomas).

# In[10]:


# Soil type count
soil_features = [x for x in new_train_data.columns if x.startswith("Soil_Type")]
new_train_data["Soil_Type_Count"] = new_train_data[soil_features].sum(axis=1)
new_test_data["Soil_Type_Count"] = new_test_data[soil_features].sum(axis=1)

# Wilderness area count
wilderness_features = [x for x in new_train_data.columns if x.startswith("Wilderness_Area")]
new_train_data["Wilderness_Area_Count"] = new_train_data[wilderness_features].sum(axis=1)
new_test_data["Wilderness_Area_Count"] = new_test_data[wilderness_features].sum(axis=1)


# **Drop features with 0 variance**

# In[11]:


# Train df
new_train_data.drop('Soil_Type7', axis=1, inplace=True)
new_train_data.drop('Soil_Type15', axis=1, inplace=True)

# Test df
new_test_data.drop('Soil_Type7', axis=1, inplace=True)
new_test_data.drop('Soil_Type15', axis=1, inplace=True)


# # Memory

# In[12]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[13]:


new_train_data=reduce_mem_usage(new_train_data)
new_test_data=reduce_mem_usage(new_test_data)


# # Pre-process data

# **Labels and features:**

# In[14]:


# Labels
y=new_train_data.Cover_Type

# Features
X=new_train_data.drop('Cover_Type', axis=1)


# **Scale data**

# In[15]:


scaler = StandardScaler()
X=scaler.fit_transform(X)
test_data_preprocessed = scaler.transform(new_test_data)


# **Label encoding**

# In[16]:


# Encode labels to lie in range 0 to 5
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# **Save memory**

# In[17]:


del train_data, test_data, scaler
del pseudo_label_df, new_train_data, new_test_data
del mask_features, hillshades
del soil_features,wilderness_features


# # Neural network

# In[18]:


# Define model
def build_model():
    model = keras.Sequential([

        # hidden layer 1
        layers.Dense(units=256, activation='relu', input_shape=[X.shape[1]], kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.3),

        # hidden layer 2
        layers.Dense(units=256, activation='relu', kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.3),

        # hidden layer 3
        layers.Dense(units=128, activation='relu', kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.2),
        
        # hidden layer 4
        layers.Dense(units=64, activation='relu', kernel_initializer='lecun_normal'),
        layers.Dropout(rate=0.2),

        # output layer
        layers.Dense(units=6, activation='softmax')
    ])
    
    # Define loss, optimizer and metric
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model


# **Callbacks**

# In[19]:


# Define early stopping callback on validation loss
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)

# Reduce learning rate when validation loss plateaus
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5
)


# # Cross validation

# Credit: [Gulshan](https://www.kaggle.com/gulshanmishra/tps-dec-21-tensorflow-nn-feature-engineering).

# In[20]:


FOLDS = 8
EPOCHS = 100
BATCH_SIZE = 250

test_preds = np.zeros((1, 1))
scores = []

cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    # Start timer
    start = time.time()
    
    # get training and validation sets
    X_train, X_valid = X[train_idx], X[val_idx]
    y_train, y_valid = y[train_idx], y[val_idx]

    # Build and train model on tpu
    model = build_model()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=False
    )

    # Make predictions and get measure accuracy
    y_pred = np.argmax(model.predict(X_valid), axis=1)
    score = accuracy_score(y_valid, y_pred)
    scores.append(score)
    
    # Store predictions
    test_preds = test_preds + model.predict(test_data_preprocessed)
    
    # Stop timer
    stop = time.time()
    
    # Print accuracy and time
    print(f"Fold {fold} - Accuracy: {score}, Time: {round((stop - start)/60,1)} mins")
    
print('')
print(f"Mean Accuracy: {np.mean(scores)}")


# **Soft voting**

# In[21]:


# Soft voting to ensemble predictions
test_preds = np.argmax(test_preds, axis=1)

# Recover class labels
pred_classes = encoder.inverse_transform(test_preds)


# # Submission

# In[22]:


# Save new predictions to df
new_test_preds_df=pd.DataFrame({'Id': new_test_data_index, 
                                'Cover_Type': pred_classes})

# Concatenate with pseudolabels
final_preds=pd.concat([new_test_preds_df, pseudo_label_preds_df])

# Sort by id
final_preds=final_preds.sort_values(by='Id', ascending=True)

# Check format
final_preds.head(10)


# In[23]:


# Save to csv
final_preds.to_csv('submission.csv', index=False)

