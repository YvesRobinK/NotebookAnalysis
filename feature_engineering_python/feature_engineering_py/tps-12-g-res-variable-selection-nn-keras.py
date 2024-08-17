#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import numpy as np
import pandas as pd
import datatable as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

from warnings import filterwarnings
filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# <img src="https://i.ibb.co/PWvpT9F/header.png" alt="header" border="0" width=800 height=300>

# # Introduction

# <div style="font-size:110%"> 
# <p>Hi,</p>
# <p>while I'm still enjoying my deep-learning adventure, reading and learning a ton of stuff - I decided it's time to implement different kinds of networks with this month competition. This Notebook is all about <b>Gated Residual (GRN)</b> and <b>Variable Selection Networks (VSN)</b> based on the <a href="https://keras.io/examples/structured_data/classification_with_grn_and_vsn/">keras.io implementation</a>.</p>
# <p>The main idea here is to use the GRN's gating mechanism to (soft-) filter out less important features to use the network's learning capacity on the more salient features. The following steps can be used to describe the inner workings on a higher level and to provide some intuition:</p>
# <ol>
#     <li>Create a feature embedding (linear projection) as model input</li>
#     <li>Apply GRN to each feature individually (filter out less important imput per feature)</li>
#     <li>Apply GRN to concatenated features (get features weights, importance of one feature compared to the complete feature space</li>
#     <li>Create weighted sum of (2) & (3) as VSN output</li>
#     <li>Create final prediction with the usual dense(softmax) layer</li>
# </ol>
# <p><em>The drawing below might help as well.</em></p>
# </div>

# <blockquote><img src="https://i.ibb.co/sqhrnfV/GRN-VSN.png" alt="GRN-VSN" border="0">

# <div style="font-size:110%">
# <p>Feel free to take a look at my other notebooks, covering some different ideas and architectures:
#     <li><a href="https://www.kaggle.com/mlanhenke/tps-12-simple-nn-baseline-keras">Simple NN Baseline</a></li>
#     <li><a href="https://www.kaggle.com/mlanhenke/tps-12-deep-wide-nn-keras">Deep & Wide NN </a></li>
#     <li><a href="https://www.kaggle.com/mlanhenke/tps-12-deep-cross-nn-keras">Deep & Cross NN</a></li>
#     <li><a href="https://www.kaggle.com/mlanhenke/tps-12-denoising-autoencoder-nn-keras">Deepstack Denoising Autoencoder</a></li>
#     <li><a href="https://www.kaggle.com/mlanhenke/tps-12-bottleneck-autoencoder-nn-keras">Bottleneck Autoencoder</a></li>
# </p>
#     
# <em>Thank you very much for taking some time to read my notebook. Please leave an upvote if you find any of this information useful.</em>
# </div>

# # Import & Prepare Data

# In[2]:


def feature_engineering(df):
    df['Aspect'][df['Aspect'] < 0] += 360 
    df['Aspect'][df['Aspect'] > 359] -= 360
    
    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
    
    features_Hillshade = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    df["hillshade_mean"] = df[features_Hillshade].mean(axis = 1)
    df['hillshade_amp'] = df[features_Hillshade].max(axis = 1) - df[features_Hillshade].min(axis = 1)
    
    df["ecldn_dist_hydrlgy"] = (df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2) ** 0.5
    df["mnhttn_dist_hydrlgy"] = np.abs(df["Horizontal_Distance_To_Hydrology"]) + np.abs(df["Vertical_Distance_To_Hydrology"])
    df['binned_elevation'] = [np.floor(v/50.0) for v in df['Elevation']]
    df['highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    
    soil_features = [x for x in df.columns if x.startswith("Soil_Type")]
    df["soil_type_count"] = df[soil_features].sum(axis=1)
    
    wilderness_features = [x for x in df.columns if x.startswith("Wilderness_Area")]
    df["wilderness_area_count"] = df[wilderness_features].sum(axis = 1)
    
    df['soil_Type12_32'] = df['Soil_Type32'] + df['Soil_Type12']
    df['soil_Type23_22_32_33'] = df['Soil_Type23'] + df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']
    
    df['Horizontal_Distance_To_Roadways'][df['Horizontal_Distance_To_Roadways'] < 0] = 0
    df['horizontal_Distance_To_Roadways_Log'] = [np.log(v+1) for v in df['Horizontal_Distance_To_Roadways']]
    df['Horizontal_Distance_To_Fire_Points'][df['Horizontal_Distance_To_Fire_Points'] < 0] = 0
    df['horizontal_Distance_To_Fire_Points_Log'] = [np.log(v+1) for v in df['Horizontal_Distance_To_Fire_Points']]
    return df


# In[3]:


# import train & test data
df_train = dt.fread('../input/tabular-playground-series-dec-2021/train.csv').to_pandas()
df_test = dt.fread('../input/tabular-playground-series-dec-2021/test.csv').to_pandas()
df_pseudo = dt.fread('../input/tps12-pseudolabels/tps12-pseudolabels_v2.csv').to_pandas()

df_train = pd.concat([df_train, df_pseudo], axis=0)

sample_submission = pd.read_csv('../input/tabular-playground-series-dec-2021/sample_submission.csv')

# drop underrepresented class
df_train = df_train[df_train['Cover_Type'] != 5]

# apply feature-engineering
# thanks to https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293373
# thanks to https://www.kaggle.com/teckmengwong/dcnv2-softmaxclassification#Feature-Engineering
# df_train = feature_engineering(df_train)
# df_test = feature_engineering(df_test)

# split dataframes for later modeling
X = df_train.drop(columns=['Id','Cover_Type','Soil_Type7','Soil_Type15','Soil_Type1']).copy()
y = df_train['Cover_Type'].copy()

X_test = df_test.drop(columns=['Id','Soil_Type7','Soil_Type15','Soil_Type1']).copy()

# create label-encoded one-hot-vector for softmax, mutliclass classification
le = LabelEncoder()
target = keras.utils.to_categorical(le.fit_transform(y))

del df_train, df_test
gc.collect()

print(X.shape, y.shape, target.shape, X_test.shape)


# In[4]:


# define helper functions
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Seed set to: {seed}")

def plot_eval_results(scores, n_splits):
    cols = 5
    rows = int(np.ceil(n_splits/cols))
    
    fig, ax = plt.subplots(rows, cols, tight_layout=True, figsize=(20,2.5))
    ax = ax.flatten()

    for fold in range(len(scores)):
        df_eval = pd.DataFrame({'train_loss': scores[fold]['loss'], 'valid_loss': scores[fold]['val_loss']})

        sns.lineplot(
            x=df_eval.index,
            y=df_eval['train_loss'],
            label='train_loss',
            ax=ax[fold]
        )

        sns.lineplot(
            x=df_eval.index,
            y=df_eval['valid_loss'],
            label='valid_loss',
            ax=ax[fold]
        )

        ax[fold].set_ylabel('')

    sns.despine()

def plot_cm(cm):
    metrics = {
        'accuracy': cm / cm.sum(),
        'recall' : cm / cm.sum(axis=1),
        'precision': cm / cm.sum(axis=0)
    }
    
    fig, ax = plt.subplots(1,3, tight_layout=True, figsize=(15,5))
    ax = ax.flatten()

    mask = (np.eye(cm.shape[0]) == 0) * 1

    for idx, (name, matrix) in enumerate(metrics.items()):

        ax[idx].set_title(name)

        sns.heatmap(
            data=matrix,
            cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True),
            cbar=False,
            mask=mask,
            lw=0.25,
            annot=True,
            fmt='.2f',
            ax=ax[idx]
        )
    sns.despine()


# # Model Setup

# In[5]:


# define callbacks
lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=5, 
    verbose=True
)

es = keras.callbacks.EarlyStopping(
    monitor="val_acc", 
    patience=10, 
    verbose=True, 
    mode="max", 
    restore_best_weights=True
)


# In[6]:


class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.grns = list()
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs


# In[7]:


def create_model_inputs():
    inputs = {}
    for feature_name in X.columns:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )
    return inputs

def encode_inputs(inputs, encoding_size):
    encoded_features = []
    for col in range(inputs.shape[1]):
        encoded_feature = tf.expand_dims(inputs[:, col], -1)
        encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features

def create_model(encoding_size, dropout_rate=0.15):
    inputs = layers.Input(len(X.columns))
    feature_list = encode_inputs(inputs, encoding_size)
    num_features = len(feature_list)

    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )

    outputs = layers.Dense(units=target.shape[-1], activation="softmax")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# # Training

# In[8]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tf_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except:
    tf_strategy = tf.distribute.get_strategy()
    print(f"Running on {tf_strategy.num_replicas_in_sync} replicas")
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[9]:


seed = 2021
set_seed(seed)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

predictions = []
oof_preds = {'y_valid': list(), 'y_hat': list()}
scores_nn = {fold:None for fold in range(cv.n_splits)}

for fold, (idx_train, idx_valid) in enumerate(cv.split(X,y)):
    X_train, y_train = X.iloc[idx_train], target[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], target[idx_valid]
    
    scl = RobustScaler()
    X_train = scl.fit_transform(X_train)
    X_valid = scl.transform(X_valid)
    
    with tf_strategy.scope():
        model = create_model(encoding_size=128)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['acc']
        )
        
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=90,
        batch_size=4096,
        shuffle=True,
        verbose=False,
        callbacks=[lr,es]
    )
    
    scores_nn[fold] = history.history
    
    oof_preds['y_valid'].extend(y.iloc[idx_valid])
    oof_preds['y_hat'].extend(model.predict(X_valid, batch_size=4096))
    
    prediction = model.predict(scl.transform(X_test), batch_size=4096) 
    predictions.append(prediction)
    
    del model, prediction
    gc.collect()
    K.clear_session()
    
    print('_'*65)
    print(f"Fold {fold+1} || Min Val Loss: {np.min(scores_nn[fold]['val_loss'])}")
    print('_'*65)
    
print('_'*65)
overall_score = [np.min(scores_nn[fold]['val_loss']) for fold in range(cv.n_splits)]
print(f"Overall Mean Validation Loss: {np.mean(overall_score)}")


# # Evaluation & Submission

# In[10]:


plot_eval_results(scores_nn, cv.n_splits)


# In[11]:


# prepare oof_predictions
oof_y_true = np.array(oof_preds['y_valid'])
oof_y_hat = le.inverse_transform(np.argmax(oof_preds['y_hat'], axis=1))

# create confusion matrix, calculate accuracy, recall & precision
cm = pd.DataFrame(data=confusion_matrix(oof_y_true, oof_y_hat, labels=le.classes_), index=le.classes_, columns=le.classes_)
plot_cm(cm)


# In[12]:


#create final prediction, inverse labels to original classes
final_predictions = le.inverse_transform(np.argmax(sum(predictions), axis=1))

sample_submission['Cover_Type'] = final_predictions
sample_submission.to_csv('./baseline_nn.csv', index=False)

sns.countplot(final_predictions)
sns.despine()


# In[13]:


sample_submission.head()

