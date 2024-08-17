#!/usr/bin/env python
# coding: utf-8

# I developed a quick Deep Learning forecast notebook using TensorFlow and Keras. 
# I divided the datase into training and test to check the possible SMAPE such approach would give.
# For UPDRS_1 and 3 I thought th approach seemed promissing. Sharing here if anyone is interested. :)

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)


# # Auxiliary Functions

# In[2]:


def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = dem != 0
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

def prepare_data(train_dataset, test_dataset, label):
    
    # Break into target and features
    train_dataset = train_dataset.dropna(subset=[label])
    test_dataset = test_dataset.dropna(subset=[label])

    train_features = train_dataset.drop(target, axis=1).copy()
    test_features = test_dataset.drop(target, axis=1).copy()

    train_labels = train_dataset[label]
    test_labels = test_dataset[label]
    
    # Treat Na, 0 and 1s
    for c in train_features.columns:
        m = train_features[c].mean()
        train_features[c] = train_features[c].fillna(m)

    for c in test_features.columns:
        m = test_features[c].mean()
        test_features[c] = test_features[c].fillna(m)
        
    return train_features, test_features, train_labels, test_labels


# # Read Data

# In[3]:


train = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_clinical_data.csv")
pro = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_proteins.csv")
pep = pd.read_csv("/kaggle/input/amp-parkinsons-disease-progression-prediction/train_peptides.csv")


# # Feature Engineering

# In[4]:


# Dropping because I am not using it right now
train = train.drop(['upd23b_clinical_state_on_medication', 'patient_id'], axis=1)


# In[5]:


# We create a feature of peptides / proteins, to see the amount of present in each protein

pep_pro = pd.merge(pro, pep, on=['visit_id', 'visit_month', 'patient_id', 'UniProt'])
pep_pro['pep_per_pro'] = pep_pro['PeptideAbundance'] / pep_pro['NPX']


# In[6]:


pep_pro = pep_pro.drop(['patient_id', 'visit_month'], axis=1).pivot(index=['visit_id'], 
                                                                    columns=['Peptide'], 
                                                                    values=['pep_per_pro'])


# In[7]:


pep_pro.columns = pep_pro.columns.droplevel()
pep_pro = pep_pro.reset_index()
pep_pro.head()


# In[8]:


# Merging dataframes to get final dataset for modelling

df = pd.merge(train, pep_pro, on="visit_id", how="left")
df = df.set_index('visit_id')
df.head()


# # Modelling with Deep Learning
# 
# We will iterate and create one deep learning model per target (updrs). 
# 80% training of the dataset is used to train the model and 20% to test it.
# Added 6 intermediate layers in my neural netwotk, each with 969 units because that is how many features I have.

# In[9]:


target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

results = {}

for label in target:

    train_features, test_features, train_labels, test_labels = prepare_data(train_dataset, 
                                                                            test_dataset, 
                                                                            label)    
    # Normalization - good to have before using deep learning
    features = np.array(train_features)
    feat_normalizer = layers.Normalization(axis=-1)
    feat_normalizer.adapt(features)

    # Model Definitoon
    updrs_model = tf.keras.Sequential([
        feat_normalizer,
        layers.Dense(969, activation='relu'),
        layers.Dense(969, activation='relu'),
        layers.Dense(969, activation='relu'),
        layers.Dense(969, activation='relu'),
        layers.Dense(900, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(units=1)
    ])

    # Princting the model layout just for fun
    print(updrs_model.summary())

    # Training
    updrs_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    history = updrs_model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.2)

    # Predict
    y = updrs_model.predict(test_features)
    temp = pd.DataFrame({"y_test": test_labels, 'y_pred': y.tolist()})
    temp['y_pred'] = temp["y_pred"].apply(lambda x: x[0])

    # Smape
    results[label] = smape(temp["y_test"], temp['y_pred'])
    print(label, results[label] )


# # Show final SMAPE results

# In[10]:


results


# In[ ]:




