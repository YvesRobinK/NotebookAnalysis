#!/usr/bin/env python
# coding: utf-8

# In this notebook, I demonstrate how to use the multiclass focal loss that should help you score better with such imbalanced classes. The focal loss function is from https://github.com/artemmavrin/focal-loss/blob/master/docs/source/index.rst
# 
# The focal loss is a loss that has been devised for object detection problems where the background is more prominent than the objects to be detected. 
# 
# ![](https://github.com/Atomwh/FocalLoss_Keras/raw/master/images/fig1-focal%20loss%20results.png)
# 
# As you increase the gamma value, you put more emphasis on hard to classify examples. There is clearly a trade-off for this (high gamma values can be detrimental), but overall if you set the right value it should perform much better than using other tricks for imbalanced data.
# 
# This notebook owes quite a lot of ideas from "TPSDEC21-01-Keras Quickstart" (https://www.kaggle.com/ambrosm/tpsdec21-01-keras-quickstart) by @ambrosm please consider upvoting also his work.
# 
# It also implements the feature engineering suggested by @aguschin (see my post https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/291839 for all the references).

# In[1]:


get_ipython().system('pip install git+https://github.com/artemmavrin/focal-loss.git')


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from warnings import filterwarnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold

filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# In[3]:


from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Input, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa


# In[4]:


def plot_keras_history(history, measures):
    """
    history: Keras training history
    measures = list of names of measures
    """
    rows = len(measures) // 2 + len(measures) % 2
    fig, panels = plt.subplots(rows, 2, figsize=(15, 5))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    try:
        panels = [item for sublist in panels for item in sublist]
    except:
        pass
    for k, measure in enumerate(measures):
        panel = panels[k]
        panel.set_title(measure + ' history')
        panel.plot(history.epoch, history.history[measure], label="Train "+measure)
        panel.plot(history.epoch, history.history["val_"+measure], label="Validation "+measure)
        panel.set(xlabel='epochs', ylabel=measure)
        panel.legend()
        
    plt.show(fig)


# In[5]:


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


# In[6]:


train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
submission = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")


# In[7]:


# source: https://www.kaggle.com/remekkinas/tps-12-nn-tpu-pseudolabeling-0-95661
pseudolabels = pd.read_csv("../input/tps12-pseudolabels/tps12-pseudolabels_v2.csv")


# In[8]:


print("The target class distribution:")
print((train.groupby('Cover_Type').Id.nunique() / len(train)).apply(lambda p: f"{p:.3%}"))


# In[9]:


# Droping Cover_Type 5 label, since there is only one instance of it
train = train[train.Cover_Type != 5]


# In[10]:


# remove unuseful features
train = train.drop([ 'Soil_Type7', 'Soil_Type15'], axis=1)
pseudolabels = pseudolabels.drop([ 'Soil_Type7', 'Soil_Type15'], axis=1)
test = test.drop(['Soil_Type7', 'Soil_Type15'], axis=1)

# extra feature engineering
def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

def fe(df):
    df['EHiElv'] = df['Horizontal_Distance_To_Roadways'] * df['Elevation']
    df['EViElv'] = df['Vertical_Distance_To_Hydrology'] * df['Elevation']
    df['Aspect2'] = df.Aspect.map(r)
    ### source: https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293373
    df["Aspect"][df["Aspect"] < 0] += 360
    df["Aspect"][df["Aspect"] > 359] -= 360
    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
    ########
    df['Highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Euclidean_Distance_to_Hydrolody'] = (df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)**0.5
    df['Manhattan_Distance_to_Hydrolody'] = df['Horizontal_Distance_To_Hydrology'] + df['Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    df['Hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)
    return df

train = fe(train)
test = fe(test)
pseudolabels = fe(pseudolabels)

# Summed features pointed out by @craigmthomas (https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/292823)
soil_features = [x for x in train.columns if x.startswith("Soil_Type")]
wilderness_features = [x for x in train.columns if x.startswith("Wilderness_Area")]

train["soil_type_count"] = train[soil_features].sum(axis=1)
pseudolabels["soil_type_count"] = pseudolabels[soil_features].sum(axis=1)
test["soil_type_count"] = test[soil_features].sum(axis=1)

train["wilderness_area_count"] = train[wilderness_features].sum(axis=1)
pseudolabels["wilderness_area_count"] = pseudolabels[wilderness_features].sum(axis=1)
test["wilderness_area_count"] = test[wilderness_features].sum(axis=1)


# In[11]:


train = reduce_mem_usage(train)
pseudolabels = reduce_mem_usage(pseudolabels)
original_len = len(train)
train = pd.concat([train, pseudolabels], axis=0)


# In[12]:


y = train.Cover_Type.values - 1
X = train.drop("Cover_Type", axis=1).set_index("Id").values.astype(np.float32)
Xt = test.set_index("Id").values.astype(np.float32)


# In[13]:


import gc
del([train, test, pseudolabels])
_ = [gc.collect() for i in range(5)]


# In[14]:


le = LabelEncoder()
target = le.fit_transform(y)

_, classes_num = np.unique(target, return_counts=True)


# In[15]:


### create baseline-model
def get_model(layers=[8], targets=7, dropout_rate=0.0, skip_layers=True, 
              batchnorm=True, activation='selu', kernel_initializer="lecun_normal"):
    
    inputs_sequence = Input(shape=(X.shape[1]))
    x = Flatten()(inputs_sequence)

    skips = list()
    for layer, nodes in enumerate(layers):
        x = Dense(nodes, kernel_initializer=kernel_initializer, activation=activation)(x)
        if batchnorm is True:
            x = BatchNormalization()(x)
        if layer != (len(layers) - 1):
            if dropout_rate > 0:
                x = Dropout(rate=dropout_rate)(x)
            skips.append(x)
    
    if skip_layers is True:
        x = Concatenate(axis=1)([x] + skips)
    else:
        del(skips)
        
    output_class = Dense(targets, activation='softmax', 
                         kernel_regularizer=tf.keras.regularizers.l2(l2=0.03))(x)

    model = Model(inputs=inputs_sequence, outputs=output_class)
    
    return model


# In[16]:


dnn_params = {'layers': [128, 64, 64, 64], 
              'batchnorm': True, 
              'skip_layers': True, 
              'targets': len(le.classes_)}

model = get_model(**dnn_params)
model.summary()


# In[17]:


plot_model(
    model, 
    to_file='baseline.png', 
    show_shapes=True,
    show_layer_names=True
)


# In[18]:


try:
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    # instantiate a distribution strategy
    tf_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except:
    tf_strategy = tf.distribute.get_strategy()
    print(f"Running on {tf_strategy.num_replicas_in_sync} replicas")
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[19]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

### define callbacks
early_stopping = EarlyStopping(
    monitor='val_acc', 
    min_delta=0, 
    patience=10, 
    verbose=0,
    mode='max', 
    baseline=None, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_acc', 
    factor=0.5,
    patience=5,
    mode='max'
)


# In[20]:


N_FOLDS = 20

### cross-validation 
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1)

predictions = np.zeros((len(Xt), len(le.classes_)))
oof = np.zeros((original_len, len(le.classes_)))
scores = list()

with tf_strategy.scope():
    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
        
        idx_valid = idx_valid[idx_valid<original_len]
        X_train, y_train = X[idx_train, :], target[idx_train]
        X_valid, y_valid = X[idx_valid, :], target[idx_valid]
        
        ss = RobustScaler()
        X_train = ss.fit_transform(X_train)
        X_valid = ss.transform(X_valid)

        model = get_model(**dnn_params)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)    
    
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), #SparseCategoricalFocalLoss(gamma=2.), tf.keras.losses.SparseCategoricalCrossentropy()
            metrics=['acc']
        )

        print('**'*20)
        print(f"Fold {fold+1} || Training")
        print('**'*20)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=1024*2,
            epochs=150,
            verbose=1,
            shuffle=True,
            callbacks=[
                early_stopping,
                reduce_lr
            ]
        )
        
        plot_keras_history(history, ['loss', 'acc'])
        
        print(f"Best training accuracy: {np.max(history.history['acc']):0.5f}")
        print(f"Best validation accuracy: {np.max(history.history['val_acc']):0.5f}")
        scores.append(np.max(history.history['val_acc']))

        oof[idx_valid] = model.predict(X_valid, batch_size=4096) 

        predictions += model.predict(ss.transform(Xt), batch_size=4096)
        
        del([X_train, y_train, X_valid, y_valid])
        gc.collect()


# In[21]:


print(f"Average cv accuracy: {np.mean(scores):0.5f} (std={np.std(scores):0.5f})")


# In[22]:


submission.Cover_Type = le.inverse_transform(np.argmax(predictions, axis=1)) + 1
submission.to_csv("submission.csv", index=False)


# In[23]:


oof = pd.DataFrame(oof, columns=[f"prob_{i}" for i in le.classes_])
oof.insert(loc=0, column='Id', value=range(len(oof)))
oof.to_csv("oof.csv", index=False)

