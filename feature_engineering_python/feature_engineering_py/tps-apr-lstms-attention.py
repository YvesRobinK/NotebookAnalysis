#!/usr/bin/env python
# coding: utf-8

# <div class="text_cell_render border-box-sizing rendered_html">
# <div style="background-color:rgba(0, 167, 255, 0.6);border-radius:5px;display:fill">
# <h1><center>Tabular Playground Series - Apr 2022</center></h1></div>
# </div>

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *

np.random.seed(42)
tf.random.set_seed(42)

train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')
subs = pd.read_csv('../input/tabular-playground-series-apr-2022/sample_submission.csv')


# ### Install Keras Self Attention

# In[2]:


get_ipython().system('pip install keras-self-attention')


# ### Preprocessing for DNN

# In[3]:


### Credits https://www.kaggle.com/code/dmitryuarov/tps-sensors-2xlstm-xgb-auc-0-976

features = train.columns.tolist()[3:]

def preprocessing(df):
    for feature in features:
        df[feature + '_lag1'] = df.groupby('sequence')[feature].shift(1)
        df.fillna(0, inplace=True)
        df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']
        
preprocessing(train)
preprocessing(test)

features = train.columns.tolist()[3:]
std_sc = StandardScaler()
train[features] = std_sc.fit_transform(train[features])
test[features] = std_sc.transform(test[features])

groups = train['sequence']
labels = train_labels['state']

train = train.drop(['sequence', 'subject', 'step'], axis=1).values
train = train.reshape(-1, 60, train.shape[-1])

test = test.drop(['sequence', 'subject', 'step'], axis=1).values
test = test.reshape(-1, 60, test.shape[-1])


# ### Model

# In[4]:


from keras_self_attention import SeqSelfAttention

def lstm_att_model():

    x_input = Input(shape=(train.shape[-2:]))
    
   
    x = Bidirectional(LSTM(512, return_sequences=True))(x_input)
    x = Bidirectional(LSTM(384, return_sequences=True))(x)
    x = SeqSelfAttention(attention_activation='sigmoid',name='attention_weight')(x)
    x = GlobalAveragePooling1D()(x)
    
    x_output = Dense(units=1, activation="sigmoid")(x)
    
    model = Model(inputs=x_input, outputs=x_output, name='alstm_model')
    
    return model

model = lstm_att_model()

plot_model(
    model, 
    to_file='Att_Model.png', 
    show_shapes=False,
    show_layer_names=True
)


# In[5]:


BATCH_SIZE = 256
VERBOSE = False
predictions, scores = [], []
k = GroupKFold(n_splits = 5)
for fold, (train_idx, val_idx) in enumerate(k.split(train, labels, groups.unique())):
    print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)

    X_train, X_val = train[train_idx], train[val_idx]
    y_train, y_val = labels.iloc[train_idx].values, labels.iloc[val_idx].values
    
    model = lstm_att_model()
    #print('model name:',model.name)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics='AUC')

    lr = ReduceLROnPlateau(monitor="val_auc", factor=0.5, 
                           patience=2, verbose=VERBOSE, mode="max")

    es = EarlyStopping(monitor="val_auc", patience=7, 
                       verbose=VERBOSE, mode="max", 
                       restore_best_weights=True)
    
    chk_point = ModelCheckpoint(f'./TPS_model_2022_{fold+1}C.h5', 
                                monitor='val_auc', verbose=VERBOSE, 
                                save_best_only=True, mode='max')
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              epochs=20,
              verbose=VERBOSE,
              batch_size=BATCH_SIZE, 
              callbacks=[lr, chk_point, es])
    
    model = load_model(f'./TPS_model_2022_{fold+1}C.h5', custom_objects=SeqSelfAttention.get_custom_objects())
    
    y_pred = model.predict(X_val, batch_size=BATCH_SIZE).squeeze()
    score = roc_auc_score(y_val, y_pred)
    scores.append(score)
    predictions.append(model.predict(test, batch_size=BATCH_SIZE).squeeze())
    print(f"Fold-{fold+1} | OOF Score: {score}")

print(f'Mean AUC on {k.n_splits} folds - {np.mean(scores)}')


# ### Submission

# In[6]:


subs["state"] = sum(predictions)/k.n_splits 
subs.to_csv('submission.csv', index=False)
subs.head()

