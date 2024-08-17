#!/usr/bin/env python
# coding: utf-8

# Inspired by the kernel [openvaccine-gru-lstm](https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm).
# 
# ### Origin of the gru-lstm hybrid models
# ### Reviews appreciated:)

# In[1]:


import warnings
warnings.filterwarnings('ignore')

#the basics
import pandas as pd, numpy as np, seaborn as sns
import math, json, os, random
from matplotlib import pyplot as plt
from tqdm import tqdm

#tensorflow basics
import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K

#for model evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold


# In[2]:


def seed_everything(seed = 34):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything()


# In[3]:


#get comp data
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


# In[4]:


#Exploring signal_to_noise and SN_filter distributions
fig, ax = plt.subplots(1, 2, figsize = (15, 5))
sns.kdeplot(train['signal_to_noise'], shade = True, ax = ax[0])
sns.countplot(train['SN_filter'], ax = ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution');


# ## Processing

# In[5]:


#target columns
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


# In[6]:


token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}


# In[7]:


def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )


# In[8]:


train_inputs = preprocess_inputs(train)
train_labels = np.array(train[target_cols].values.tolist()).transpose((0, 2, 1))


# In[9]:


def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


# In[10]:


def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))

def build_model(gru=1, seq_len=107, pred_len=68, dropout=0.4,
                embed_dim=100, hidden_dim=128, layers=3):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    hidden = tf.keras.layers.SpatialDropout1D(.2)(reshaped)  
    
    
    if gru==1:
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    elif gru==0:
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==3:
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==4:
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==5:
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    #only making predictions on the first part of each sequence
    truncated = hidden[:, :pred_len]
    
    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    #some optimizers
    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)
    
    model.compile(optimizer=adam, loss=mcrmse)
    
    return model


# # Training
# 
# **Create train/val split now so both models are trained and evaluated on the same samples:**

# In[11]:


#basic training configuration
FOLDS = 5
EPOCHS = 100
REPEATS = 1
BATCH_SIZE = 64
VERBOSE = 2
SEED = 34


# In[12]:


#get different test sets and process each
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)


# **We will use a simple learning rate callback for now:**

# In[13]:


lr_callback = tf.keras.callbacks.ReduceLROnPlateau()


# ### 1. GRU

# In[14]:


gru_histories = []
gru_private_preds = np.zeros((private_df.shape[0], 130, 5))
gru_public_preds = np.zeros((public_df.shape[0], 107, 5))

rskf = RepeatedStratifiedKFold(FOLDS, n_repeats = REPEATS, random_state = SEED)

for f, (train_index, val_index) in enumerate(rskf.split(train_inputs, train['SN_filter'])):

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'gru-{f}.h5')

    train_ = train_inputs[train_index]
    train_labs = train_labels[train_index]
    val_ = train_inputs[val_index]
    val_labs = train_labels[val_index]

    gru = build_model(gru=1)
    history = gru.fit(train_, train_labs, 
                      validation_data=(val_,val_labs),
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=[lr_callback,sv_gru],
                      verbose = VERBOSE)  

    gru_histories.append(history)

    #load best model and predict
    gru_short = build_model(gru=1, seq_len=107, pred_len=107)
    gru_short.load_weights(f'gru-{f}.h5')
    gru_public_pred = gru_short.predict(public_inputs) / FOLDS

    gru_long = build_model(gru=1, seq_len=130, pred_len=130)
    gru_long.load_weights(f'gru-{f}.h5')
    gru_private_pred = gru_long.predict(private_inputs) / FOLDS * REPEATS

    gru_public_preds += gru_public_pred
    gru_private_preds += gru_private_pred

    del gru_short, gru_long


# ### 2. LSTM

# In[15]:


lstm_histories = []
lstm_private_preds = np.zeros((private_df.shape[0], 130, 5))
lstm_public_preds = np.zeros((public_df.shape[0], 107, 5))

rskf = RepeatedStratifiedKFold(FOLDS, n_repeats = REPEATS, random_state = SEED)

for f, (train_index, val_index) in enumerate(rskf.split(train_inputs, train['SN_filter'])):

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'lstm-{f}.h5')

    train_ = train_inputs[train_index]
    train_labs = train_labels[train_index]
    val_ = train_inputs[val_index]
    val_labs = train_labels[val_index]

    lstm = build_model(gru=0)
    history = lstm.fit(
                        train_, train_labs, 
                        validation_data=(val_,val_labs),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[lr_callback,sv_gru],
                        verbose = VERBOSE)  

    lstm_histories.append(history)

    #load best model and predict
    lstm_short = build_model(gru=0, seq_len=107, pred_len=107)
    lstm_short.load_weights(f'lstm-{f}.h5')
    lstm_public_pred = lstm_short.predict(public_inputs) / FOLDS

    lstm_long = build_model(gru=0, seq_len=130, pred_len=130)
    lstm_long.load_weights(f'lstm-{f}.h5')
    lstm_private_pred = lstm_long.predict(private_inputs) / FOLDS * REPEATS

    lstm_public_preds += lstm_public_pred
    lstm_private_preds += lstm_private_pred

    del lstm_short, lstm_long


# # 3. Hyb1

# In[16]:


hyb1_histories = []
hyb1_private_preds = np.zeros((private_df.shape[0], 130, 5))
hyb1_public_preds = np.zeros((public_df.shape[0], 107, 5))

rskf = RepeatedStratifiedKFold(FOLDS, n_repeats = REPEATS, random_state = SEED)

for f, (train_index, val_index) in enumerate(rskf.split(train_inputs, train['SN_filter'])):

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'hyb1-{f}.h5')

    train_ = train_inputs[train_index]
    train_labs = train_labels[train_index]
    val_ = train_inputs[val_index]
    val_labs = train_labels[val_index]

    lstm = build_model(gru=3)
    history = lstm.fit(
                        train_, train_labs, 
                        validation_data=(val_,val_labs),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[lr_callback,sv_gru],
                        verbose = VERBOSE)  

    hyb1_histories.append(history)

    #load best model and predict
    hyb1_short = build_model(gru=3, seq_len=107, pred_len=107)
    hyb1_short.load_weights(f'hyb1-{f}.h5')
    hyb1_public_pred = hyb1_short.predict(public_inputs) / FOLDS

    hyb1_long = build_model(gru=3, seq_len=130, pred_len=130)
    hyb1_long.load_weights(f'hyb1-{f}.h5')
    hyb1_private_pred = hyb1_long.predict(private_inputs) / FOLDS * REPEATS

    hyb1_public_preds += hyb1_public_pred
    hyb1_private_preds += hyb1_private_pred

    del hyb1_short, hyb1_long


# # 4. Hyb2 

# In[17]:


hyb2_histories = []
hyb2_private_preds = np.zeros((private_df.shape[0], 130, 5))
hyb2_public_preds = np.zeros((public_df.shape[0], 107, 5))

rskf = RepeatedStratifiedKFold(FOLDS, n_repeats = REPEATS, random_state = SEED)

for f, (train_index, val_index) in enumerate(rskf.split(train_inputs, train['SN_filter'])):

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'hyb2-{f}.h5')

    train_ = train_inputs[train_index]
    train_labs = train_labels[train_index]
    val_ = train_inputs[val_index]
    val_labs = train_labels[val_index]

    lstm = build_model(gru=5)
    history = lstm.fit(
                        train_, train_labs, 
                        validation_data=(val_,val_labs),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[lr_callback,sv_gru],
                        verbose = VERBOSE)  

    hyb2_histories.append(history)

    #load best model and predict
    hyb2_short = build_model(gru=5, seq_len=107, pred_len=107)
    hyb2_short.load_weights(f'hyb2-{f}.h5')
    hyb2_public_pred = hyb2_short.predict(public_inputs) / FOLDS

    hyb2_long = build_model(gru=5, seq_len=130, pred_len=130)
    hyb2_long.load_weights(f'hyb2-{f}.h5')
    hyb2_private_pred = hyb2_long.predict(private_inputs) / FOLDS * REPEATS

    hyb2_public_preds += hyb2_public_pred
    hyb2_private_preds += hyb2_private_pred

    del hyb2_short, hyb2_long


# # Model Evaluation

# In[18]:


fig, ax = plt.subplots(1, 2, figsize = (20, 10))

for history in gru_histories:
    ax[0].plot(history.history['loss'], color='C0')
    ax[0].plot(history.history['val_loss'], color='C1')
for history in lstm_histories:
    ax[1].plot(history.history['loss'], color='C0')
    ax[1].plot(history.history['val_loss'], color='C1')

ax[0].set_title('GRU')
ax[1].set_title('LSTM')

ax[0].legend(['train', 'validation'], loc = 'upper right')
ax[1].legend(['train', 'validation'], loc = 'upper right')

ax[0].set_ylabel('MCRMSE')
ax[0].set_xlabel('Epoch')
ax[1].set_ylabel('MCRMSE')
ax[1].set_xlabel('Epoch');


# In[19]:


fig, ax = plt.subplots(1, 2, figsize = (20, 10))

for history in hyb1_histories:
    ax[0].plot(history.history['loss'], color='C0')
    ax[0].plot(history.history['val_loss'], color='C1')
for history in hyb2_histories:
    ax[1].plot(history.history['loss'], color='C0')
    ax[1].plot(history.history['val_loss'], color='C1')

ax[0].set_title('HYB1')
ax[1].set_title('HYB2')

ax[0].legend(['train', 'validation'], loc = 'upper right')
ax[1].legend(['train', 'validation'], loc = 'upper right')

ax[0].set_ylabel('MCRMSE')
ax[0].set_xlabel('Epoch')
ax[1].set_ylabel('MCRMSE')
ax[1].set_xlabel('Epoch');


# # Inference and Submission

# In[20]:


public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)


# **Now we just need to change the shape of each sample to the long format:**

# In[21]:


preds_gru = []

for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru)
preds_gru_df.head()


# **Now we do the same for the LSTM model so we can blend their predictions:**

# In[22]:


preds_lstm = []

for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm)
preds_lstm_df.head()


# For Hyb1:

# In[23]:


preds_hyb1 = []

for df, preds in [(public_df, hyb1_public_preds), (private_df, hyb1_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_hyb1.append(single_df)

preds_hyb1_df = pd.concat(preds_hyb1)
preds_hyb1_df.head()


# For hyb2:

# In[24]:


preds_hyb2 = []

for df, preds in [(public_df, hyb2_public_preds), (private_df, hyb2_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_hyb2.append(single_df)

preds_hyb2_df = pd.concat(preds_hyb2)
preds_hyb2_df.head()


# **And now we blend:**

# In[25]:


blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']
blend_preds_df['reactivity'] = 0.25*preds_gru_df['reactivity'] + 0.25*preds_lstm_df['reactivity'] + 0.25*preds_hyb1_df['reactivity'] + 0.25*preds_hyb2_df['reactivity']
blend_preds_df['deg_Mg_pH10'] = 0.25*preds_gru_df['deg_Mg_pH10'] + 0.25*preds_lstm_df['deg_Mg_pH10'] + 0.25*preds_hyb1_df['deg_Mg_pH10'] + 0.25*preds_hyb2_df['deg_Mg_pH10']
blend_preds_df['deg_pH10'] = 0.25*preds_gru_df['deg_pH10'] + 0.25*preds_lstm_df['deg_pH10'] + 0.25*preds_hyb1_df['deg_pH10'] + 0.25*preds_hyb2_df['deg_pH10']
blend_preds_df['deg_Mg_50C'] = 0.25*preds_gru_df['deg_Mg_50C'] + 0.25*preds_lstm_df['deg_Mg_50C'] + 0.25*preds_hyb1_df['deg_Mg_50C'] + 0.25*preds_hyb2_df['deg_Mg_50C'] 
blend_preds_df['deg_50C'] = 0.25*preds_gru_df['deg_50C'] + 0.25*preds_lstm_df['deg_50C'] + 0.25*preds_hyb1_df['deg_50C'] + 0.25*preds_hyb2_df['deg_50C']


# In[26]:


submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])
#sanity check
submission.head()


# In[27]:


submission.to_csv('submission.csv', index=False)
print('Submission saved')


# In[ ]:




