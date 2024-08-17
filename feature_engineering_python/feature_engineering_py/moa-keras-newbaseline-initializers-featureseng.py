#!/usr/bin/env python
# coding: utf-8

# <center><h2 style='color:red'>MoA | Keras [NewBaseLine] with Features Engineering<br>Smoothing Vs Non-Smoothing</h2></center><hr>

# ## Model Based on: <a href='https://www.kaggle.com/elcaiseri/moa-keras-multilabel-classifier-nn-starter'>MoA | Keras Multilabel Classifier NN | Starter </a> Kernel.
# 
# 
# ### What is new in this Kernel?
#  1. Features Engineering, and it contains:
# - 3 SKlearn preprocessing scaler
# - Apply Rank Gauss.
# - PCA
# - SVD <== NEW
# 
#  2. Feature Selection:
# - VarianceThreshold
# 
#  3. Clean Data:
# - Mapping Data
# - drop train['cp_type'] column
# 
#  4. Model:
# - using LeakyReLU rather than 'relu'
# - Add model smoothing
#  
# * Initialize Dense Layers with "VarianceScaling" / "TruncatedNormal" ==> ' https://keras.io/api/layers/initializers/ '
# * Monitor the loss without smoothing as well and Plot the results. (From @imeintanis comment on V5)
# 
# <hr><h4>Pls <span style='color:red'>UPVOTE</span>, if you find it useful. Feedbacks is also very much appreciated.<h4>

# In[1]:


import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow_addons as tfa

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn import preprocessing

from sklearn.decomposition import PCA, TruncatedSVD

from tqdm.notebook import tqdm

import math


# In[4]:


train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

data = train_features.append(test_features)

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')


# In[5]:


import random, os, torch
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
z= np.random.randint(0, 100, size=10)

sns.set_style("whitegrid")
plt.figure(figsize=(18, 8))
sns.distplot(train_features.iloc[:, z], bins=30, color='red', label='Test')
sns.distplot(test_features.iloc[:, z], bins=30, color='green', label='Train')
plt.legend()
plt.title('Train / Test Distribution for z Features Before Featuring Eng.')
plt.xlabel('z Features')
plt.ylabel('Frequency')
plt.show()


# In[7]:


def scaling_ss(train, test):
    features = train.columns[4:]
    scaler = preprocessing.StandardScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis = 0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features

#train_features, test_features, features = scaling_ss(train_features, test_features)


# In[8]:


def scaling_mm(train, test):
    features = train.columns[2:]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis = 0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features

#train_features, test_features, features = scaling_mm(train_features, test_features)


# In[9]:


def scaling_rs(train, test):
    features = train.columns[4:]
    scaler = preprocessing.RobustScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis = 0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features

train_features, test_features, features = scaling_rs(train_features, test_features)


# In[10]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# In[11]:


#RankGauss
for col in (GENES + CELLS):
    transformer = QuantileTransformer(n_quantiles=206,random_state=0, output_distribution="normal")
    vec_len = len(train_features[col].values)
    vec_len_test = len(test_features[col].values)
    raw_vec = train_features[col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)

    train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


# In[12]:


# GENES PCA
n_comp = 600  #<--Update

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[13]:


#CELLS PCA
n_comp = 60  #<--Update

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[14]:


# GENES SVD
n_comp = 450  #<--Update

data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
data2 = (TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(data[GENES]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'svd_G-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'svd_G-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[15]:


#CELLS SVD
n_comp = 45  #<--Update

data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
data2 = (TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))
train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

train2 = pd.DataFrame(train2, columns=[f'svd_C-{i}' for i in range(n_comp)])
test2 = pd.DataFrame(test2, columns=[f'svd_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train2), axis=1)
test_features = pd.concat((test_features, test2), axis=1)


# In[16]:


def c_squared(train, test):
    
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f'{feature}_squared'] = df[feature] ** 2
    return train, test

train_features,test_features=c_squared(train_features,test_features)


# In[17]:


def c_cubed(train, test):
    
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f'{feature}_cubed'] = df[feature] ** 3
    return train, test

train_features,test_features=c_cubed(train_features,test_features)


# In[18]:


def c_sqrt(train, test):
    
    features_c = list(train.columns[776:876])
    for df in [train, test]:
        for feature in features_c:
            df[f'{feature}_sqrt'] = df[feature] ** 0.5
    return train, test

train_features,test_features=c_cubed(train_features,test_features)


# In[19]:


def scaling_rs(train, test):
    features = train.columns[4:]
    scaler = preprocessing.RobustScaler()
    scaler.fit(pd.concat([train[features], test[features]], axis = 0))
    train[features] = scaler.transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, features

#train_features, test_features, features = scaling_rs(train_features, test_features)


# In[20]:


print(f'New Train/Test Features Dataset Contains [{train_features.shape[1]}] Features.')


# In[21]:


train_features


# In[22]:


threshold = 0.9
var_thresh = VarianceThreshold(threshold)
data = train_features.append(test_features)
data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[ : train_features.shape[0]]
test_features_transformed = data_transformed[-test_features.shape[0] : ]

train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                              columns=['sig_id','cp_type','cp_time','cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)


test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                             columns=['sig_id','cp_type','cp_time','cp_dose'])

test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)


# In[23]:


print(f'Variance Threshold Select [{train_features.shape[1]}] Features From [1836]]')


# In[24]:


train = train_features.copy()
target = train_targets.copy()
test = test_features.copy()

target = target[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
target.drop(['sig_id'], axis=1, inplace=True)

train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
train.drop(['sig_id', 'cp_type'], axis=1, inplace=True)

test.drop(['sig_id', 'cp_type'], axis=1, inplace=True)


# In[25]:


train, test, features = scaling_mm(train, test)


# In[26]:


def preprocess(df):
    df = df.copy()
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    #df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})    
    df = pd.get_dummies(df, columns=['cp_time','cp_dose'])
    return df

train = preprocess(train)
test = preprocess(test)
data = train.append(test)


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
z= np.random.randint(0, 100, size=10)

sns.set_style("whitegrid")
plt.figure(figsize=(18, 8))
sns.distplot(test.iloc[:, z], bins=30, color='red', label='Test')
sns.distplot(train.iloc[:, z], bins=30, color='green', label='Train')
plt.legend()
plt.title('Train / Test Distribution for z Features After Featuring Eng.')
plt.xlabel('z Features')
plt.ylabel('Frequency')
plt.show()


# In[28]:


train


# In[29]:


train.describe()


# In[30]:


np.mean(train.values), np.std(train.values), np.min(train.values), np.max(train.values)


# In[31]:


somthing_rate = 1e-3
P_MIN = somthing_rate
P_MAX = 1 - P_MIN

def loss_fn(yt, yp):
    yp = np.clip(yp, P_MIN, P_MAX)
    return log_loss(yt, yp, labels=[0,1])

NUM_FEATURES = train.shape[1]
NUM_FEATURES


# In[32]:


def create_model(num_columns, hidden_layers=1500, SEED=None):
    model = tf.keras.Sequential([tf.keras.layers.Input(num_columns)])
    #initializer = tf.keras.initializers.VarianceScaling(scale=2., mode='fan_in', distribution='truncated_normal', seed=SEED)#math.sqrt(6. / n) 
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.5, stddev=1., seed=SEED) 

    model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(0.4))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(hidden_layers, kernel_initializer=initializer)))
    #model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2654321))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(hidden_layers, kernel_initializer=initializer)))
    #model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.LeakyReLU())

    #============ Final Layer =================
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2678923456789))
    model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, kernel_initializer=initializer)))
    model.add(tf.keras.layers.Activation('sigmoid'))
    
    tfa_opt = tfa.optimizers.Lookahead(tfa.optimizers.AdamW(lr = 1e-2, weight_decay = 1e-5), sync_period=10)
    tf_opt = tfa.optimizers.Lookahead(tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-15), sync_period=10)
    
    model.compile(optimizer=tfa_opt, 
                  loss=BinaryCrossentropy(),
                  metrics=BinaryCrossentropy(label_smoothing=somthing_rate)
                  )
    return model


# In[33]:


# Use All feats as top feats
top_feats = [i for i in range(train.shape[1])]
print("Top feats length:",len(top_feats))


# In[34]:


mod = create_model(len(top_feats))
mod.summary()


# In[35]:


def metric(y_true, y_pred):
    metrics = []
    for _target in train_targets.columns:
        metrics.append(loss_fn(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float)))
    return np.mean(metrics)


# In[36]:


N_STARTS = 5

train_targets = target

res = train_targets.copy()
ss.loc[:, train_targets.columns] = 0
res.loc[:, train_targets.columns] = 0

historys = dict()

#tf.random.set_seed(42)
seed_everything(seed=42)
for seed in range(N_STARTS):
    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=7, random_state=seed, shuffle=True).split(train_targets, train_targets)):
        print(f"======{train_targets.values[tr].shape}========{train_targets.values[te].shape}=====")
        print(f'Seed: {seed} => Fold: {n}')
        
        checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=1e-20, patience=6, verbose=1, mode='min')
        cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 1, save_best_only = True,
                                     save_weights_only = True, mode = 'min')
        early = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience= 14, verbose = 1)
        
        model = create_model(len(top_feats), SEED=seed)
        
        history = model.fit(train.values[tr][:, top_feats],
                  train_targets.values[tr],
                  validation_data=(train.values[te][:, top_feats], train_targets.values[te]),
                  epochs=100, batch_size=128,
                  callbacks=[reduce_lr_loss, cb_checkpt, early], verbose=2
                 )
        historys[f'history_seed_{seed+1}_fold_{n+1}'] = history
        print("Model History Saved.")
        
        model.load_weights(checkpoint_path)
        test_predict = model.predict(test.values[:, top_feats])
        val_predict = model.predict(train.values[te][:, top_feats])
        
        ss.loc[:, train_targets.columns] += test_predict
        res.loc[te, train_targets.columns] += val_predict
        
        print(f'OOF Metric For SEED {seed} => FOLD {n} : {metric(train_targets.loc[te, train_targets.columns], pd.DataFrame(val_predict, columns=train_targets.columns))}')
        print('+-' * 10)


# ## Smoothing vs Non-Smoothing

# In[37]:


# Show Model loss in plots
for k,v in historys.items():
    loss = []
    val_loss = []
    loss.append(v.history['loss'][:35])
    val_loss.append(v.history['val_loss'][:35])
    
# Show Model loss in plots
for k,v in historys.items():
    bin_loss = []
    bin_val_loss = []
    bin_loss.append(v.history['binary_crossentropy'][:35])
    bin_val_loss.append(v.history['val_binary_crossentropy'][:35])
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize = (23, 6))

ax[0].plot(np.mean(bin_loss, axis=0), 'b', label='Smoothing Loss')
ax[0].plot(np.mean(bin_val_loss, axis=0), 'r--', label='Smoothing Val Loss')
ax[0].set(title=f'{somthing_rate}-Somthing Model', yscale='log', yticks=[1,1e-1,1e-2], xlabel='Epoches', ylabel='Average Logloss')
ax[0].legend()

ax[1].plot(np.mean(loss, axis=0), 'b', label='Non-Smoothing Loss')
ax[1].plot(np.mean(val_loss, axis=0), 'g--',label='Non-Smoothing Val Loss')
ax[1].set(title='Non-Somthing Model', yscale='log', yticks=[1,1e-1,1e-2], xlabel='Epoches', ylabel='Average Logloss')
ax[1].legend()


ax[2].plot(np.mean(bin_val_loss, axis=0), 'r+', label='Smoothing Val Loss')
ax[2].plot(np.mean(val_loss, axis=0), 'g*',label='Non-Smoothing Val Loss')
ax[2].set(title='Somthing vs Non-Somthing Model', yscale='log', xlabel='Epoches', ylabel='Average Logloss')
ax[2].legend()


# In[38]:


ss.loc[:, train_targets.columns] /= ((n+1) * N_STARTS)
res.loc[:, train_targets.columns] /= N_STARTS


# In[39]:


print(f'OOF Metric: {metric(train_targets, res)}')


# In[40]:


np.save('oof_keras', res)
np.save('pred_keras', ss)

ss.to_csv('submission_test.csv', index=False)


# In[41]:


ss.to_csv('submission.csv', index=False)


# Kernel still under modification.. **<span style='color:red'>Feedbacks</span>** is also very much appreciated.
# Pls **<span style='color:red'>UPVOTE</span>**, if you find it useful. 
# 

# In[ ]:




