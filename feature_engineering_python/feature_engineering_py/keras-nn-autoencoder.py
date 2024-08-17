#!/usr/bin/env python
# coding: utf-8

# I've always wanted to try autoencoders for tabular data and finally found an excuse to try them out. This kernel heavily borrows from https://www.kaggle.com/abazdyrev/keras-nn-focal-loss-experiments who did a better job formatting the data than I did in my first NN starter kernel. https://www.kaggle.com/ryches/keras-nn-starter-w-time-series-split. It takes inspiration from Christof's post here https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88740#latest-515210 and Michael Jahrer's famous Porto Seguro solution

# In[1]:


import numpy as np
import pandas as pd

train = pd.read_csv('../input/train_transaction.csv')
test = pd.read_csv('../input/test_transaction.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[2]:


useful_features = list(train.iloc[:, 3:55].columns)

y = train.sort_values('TransactionDT')['isFraud']
X = train.sort_values('TransactionDT')[useful_features]
X_test = test[useful_features]
del train, test


# In[3]:


categorical_features = [
    'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2',
    'P_emaildomain',
    'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

continuous_features = list(filter(lambda x: x not in categorical_features, X))


# In[4]:


class ContinuousFeatureConverter:
    def __init__(self, name, feature, log_transform):
        self.name = name
        self.skew = feature.skew()
        self.log_transform = log_transform
        
    def transform(self, feature):
        if self.skew > 1:
            feature = self.log_transform(feature)
        
        mean = feature.mean()
        std = feature.std()
        return (feature - mean)/(std + 1e-6)        


# In[5]:


from tqdm.autonotebook import tqdm

feature_converters = {}
continuous_features_processed = []
continuous_features_processed_test = []

for f in tqdm(continuous_features):
    feature = X[f]
    feature_test = X_test[f]
    log = lambda x: np.log10(x + 1 - min(0, x.min()))
    converter = ContinuousFeatureConverter(f, feature, log)
    feature_converters[f] = converter
    continuous_features_processed.append(converter.transform(feature))
    continuous_features_processed_test.append(converter.transform(feature_test))
    
continuous_train = pd.DataFrame({s.name: s for s in continuous_features_processed}).astype(np.float32)
continuous_test = pd.DataFrame({s.name: s for s in continuous_features_processed_test}).astype(np.float32)


# In[6]:


continuous_train['isna_sum'] = continuous_train.isna().sum(axis=1)
continuous_test['isna_sum'] = continuous_test.isna().sum(axis=1)

continuous_train['isna_sum'] = (continuous_train['isna_sum'] - continuous_train['isna_sum'].mean())/continuous_train['isna_sum'].std()
continuous_test['isna_sum'] = (continuous_test['isna_sum'] - continuous_test['isna_sum'].mean())/continuous_test['isna_sum'].std()


# In[7]:


isna_columns = []
for column in tqdm(continuous_features):
    isna = continuous_train[column].isna()
    if isna.mean() > 0.:
        continuous_train[column + '_isna'] = isna.astype(int)
        continuous_test[column + '_isna'] = continuous_test[column].isna().astype(int)
        isna_columns.append(column)
        
continuous_train = continuous_train.fillna(0.)
continuous_test = continuous_test.fillna(0.)


# In[8]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm.autonotebook import tqdm

def categorical_encode(df_train, df_test, categorical_features, n_values=140):
    df_train = df_train[categorical_features].astype(str)
    df_test = df_test[categorical_features].astype(str)
    
    categories = []
    for column in tqdm(categorical_features):
        categories.append(list(df_train[column].value_counts().iloc[: n_values - 1].index) + ['Other'])
        values2use = categories[-1]
        df_train[column] = df_train[column].apply(lambda x: x if x in values2use else 'Other')
        df_test[column] = df_test[column].apply(lambda x: x if x in values2use else 'Other')
        
    
    ohe = OneHotEncoder(categories=categories)
    ohe.fit(pd.concat([df_train, df_test]))
    df_train = pd.DataFrame(ohe.transform(df_train).toarray()).astype(np.float16)
    df_test = pd.DataFrame(ohe.transform(df_test).toarray()).astype(np.float16)
    return df_train, df_test


# In[9]:


for feat in categorical_features:
    print(X[feat].nunique())


# In[10]:


train_categorical, test_categorical = categorical_encode(X, X_test, categorical_features)


# In[11]:


num_shape = continuous_train.shape[1]
cat_shape = train_categorical.shape[1]


# In[12]:


X = pd.concat([continuous_train, train_categorical], axis=1)
del continuous_train, train_categorical
X_test = pd.concat([continuous_test, test_categorical], axis=1)
del continuous_test, test_categorical


# In[13]:


test_rows = X_test.shape[0]


# In[14]:


X = pd.concat([X, X_test], axis = 0)


# In[15]:


del X_test


# In[16]:


import keras
import random
import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from keras.utils.generic_utils import get_custom_objects
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

np.random.seed(42) # NumPy
random.seed(42) # Python
tf.set_random_seed(42) # Tensorflow


# In[17]:


# Compatible with tensorflow backend
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})
get_custom_objects().update({'focal_loss_fn': focal_loss()})


# In[18]:


from keras.layers import concatenate


# What we will do here is construct a simple autoencoder that will take in our noised numeric and categorical features, concatenate them and then pass them through several dense layers that will then try to predict our original unnoised numeric and categorical features. What this will do is in essence try to learn the relationships between the features and which features should co-occur. 

# In[19]:


K.clear_session()
from keras.optimizers import Adam
def create_model():
    num_inp = Input(shape=(num_shape,))
    cat_inp = Input(shape=(cat_shape,))
    inps = concatenate([num_inp, cat_inp])
    x = Dense(512, activation=custom_gelu)(inps)
    x = Dense(256, activation=custom_gelu)(x)
    x = Dense(512, activation = custom_gelu)(x)
    x = Dropout(.2)(x)
    cat_out = Dense(cat_shape, activation = "linear")(x)
    num_out = Dense(num_shape, activation = "linear")(x)
    model = Model(inputs=[num_inp, cat_inp], outputs=[num_out, cat_out])
    model.compile(
        optimizer=Adam(.05, clipnorm = 1, clipvalue = 1),
        loss=["mse", "mse"]
    )
    return model


# In[20]:


model_mse = create_model()


# In[21]:


model_mse.summary()


# Now we need to invent some realistic noise. As Michael noted in his post he used something he called swap noise. What this is doing is swapping a columns values with other possible values from that column a certain percentage of the time. For example say there is a feature like TransactionAMT. If we used swap noise on that column it would swap 15% of the rows of the TransactionAMT column with other possible values (like swapping 20 for 400, etc.). The model would then see 400 was swapped in and all of the other features around it and it would try to learn that 20 was the real original value and try to correct the various errors we have introduced into the input.  

# In[22]:


def inputSwapNoise(arr, p):
    n, m = arr.shape
    idx = range(n)
    swap_n = round(n*p)
    for i in range(m):
        col_vals = np.random.permutation(arr[:, i]) # change the order of the row
        swap_idx = np.random.choice(idx, size= swap_n) # choose row
        arr[swap_idx, i] = np.random.choice(col_vals, size = swap_n) # n*p row and change it 
    return arr


# We will create a small generator so that we can continuously do this swapping and create new samples for the model to see

# In[23]:


def auto_generator(X, swap_rate, batch_size):
    indexes = np.arange(X.shape[0])
    while True:
        np.random.shuffle(indexes)
        num_X = X[indexes[:batch_size], :num_shape] 
        num_y = inputSwapNoise(num_X, swap_rate)
        cat_X = X[indexes[:batch_size], num_shape:] 
        cat_y = inputSwapNoise(cat_X, swap_rate)
        yield [num_y, cat_y], [num_X, cat_X]


# In[24]:


batch_size = 2048


# In[25]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler
auto_ckpt = ModelCheckpoint("ae.model", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)


# In[26]:


from keras import backend as K


class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))
warm_up_lr = WarmUpLearningRateScheduler(400, init_lr=0.005)


# In[27]:


import gc
gc.collect()


# Now we will train the autoencoder using our generator for several epochs

# In[28]:


gc.collect()
epochs = 10
train_gen = auto_generator(X.values, .15, batch_size)
hist = model_mse.fit_generator(train_gen, steps_per_epoch=len(X)//batch_size, epochs=epochs,
                           verbose=1, workers=-1, 
                           use_multiprocessing=True,
                              callbacks=[auto_ckpt, warm_up_lr])


# In[29]:


del train_gen
gc.collect()
model_mse.load_weights("ae.model")


# Now we will freeze the layers of the autoencoder

# In[30]:


for layer in model_mse.layers:
    layer.trainable = False
model_mse.compile(
    optimizer="adam",
    loss=["mse", "mse"]
)


# In[31]:


model_mse.summary()


# Next we will make a new model that branches off the previous one. This will take in non-noisy inputs and pass them through the encoding part of the autoencoder and then concatenated all of the middle layers of the encoder and then we will train our classifier based on the features that concatenated encoder outputs. 

# In[32]:


def make_model(loss_fn):
    x1 = model_mse.layers[3].output
    x2 = model_mse.layers[4].output
    x3 = model_mse.layers[5].output
    x_conc = concatenate([x1,x2,x3])
    x = Dropout(.5)(x_conc)
    x = Dense(500, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.5)(x)
    x = Dense(1, activation = 'sigmoid')(x)
    model = Model([model_mse.layers[0].input, model_mse.layers[1].input], x)
    model.compile(
        optimizer="adam",
        loss=[loss_fn]
    )
    return model


# We will train one with binary crossentropy and another with focal loss just like the previous kernel

# In[33]:


fraud_model = make_model("binary_crossentropy")
fraud_focal_model = make_model("focal_loss_fn")


# In the autoencoder we were able to take advantage of being able to train on both the train and test set because the autoencoder was trying to guess inputs rather than our target. No we will split our test set back out because we will be training on those targets for the second phase and we dont have that information for the test set

# In[34]:


X_test = X.iloc[-test_rows:, :]
X = X.iloc[:-test_rows, :]


# In[35]:


import gc
gc.collect()


# In[36]:


split_ind = int(X.shape[0]*0.8)

X_tr = X.iloc[:split_ind]
X_val = X.iloc[split_ind:]

y_tr = y.iloc[:split_ind]
y_val = y.iloc[split_ind:]

del X


# In[37]:


from keras.callbacks import ModelCheckpoint
ckpt = ModelCheckpoint("best_fraud.model", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)


# In[38]:


gc.collect()


# In[39]:


fraud_model.fit([X_tr.iloc[:, :num_shape], X_tr.iloc[:, num_shape:]], y_tr, epochs=100, batch_size=2048, 
                validation_data = ([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], y_val),
               callbacks=[ckpt], verbose = 2)


# In[40]:


valid_preds = fraud_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = 8000, verbose = True)
roc_auc_score(y_val, valid_preds)


# In[41]:


fraud_model.load_weights("best_fraud.model")
valid_preds = fraud_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = 8000, verbose = True)
roc_auc_score(y_val, valid_preds)


# In[42]:


ckpt2 = ModelCheckpoint("best_fraud_focal.model", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
fraud_focal_model.fit([X_tr.iloc[:, :num_shape], X_tr.iloc[:, num_shape:]], y_tr, epochs=100, batch_size=2048, 
                validation_data = ([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], y_val),
               callbacks=[ckpt2], verbose = 2)


# In[43]:


fraud_model.load_weights("best_fraud.model")


# In[44]:


fraud_focal_model.load_weights("best_fraud_focal.model")


# In[45]:


valid_preds = fraud_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = 8000, verbose = True)
roc_auc_score(y_val, valid_preds)


# In[46]:


from scipy.stats import rankdata, spearmanr
valid_preds = fraud_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = 8000, verbose = True)
valid_preds2 = fraud_focal_model.predict([X_val.iloc[:, :num_shape], X_val.iloc[:, num_shape:]], batch_size = 8000, verbose = True)
score = roc_auc_score(y_val, valid_preds)
score2 = roc_auc_score(y_val, valid_preds2)
score_avg = roc_auc_score(y_val, (.5*valid_preds) + (.5*valid_preds2))
print(score)
print(score2)
print(score_avg)
print('Rank averaging: ', roc_auc_score(y_val, rankdata(valid_preds, method='dense') + rankdata(valid_preds2, method='dense')))


# In[47]:


X_tr = pd.concat([X_tr, X_val, X_val, X_val, X_val], axis = 0)
y_tr = pd.concat([y_tr, y_val, y_val, y_val, y_val], axis = 0)


# In[48]:


fraud_model.fit([X_tr.iloc[:, :num_shape], X_tr.iloc[:, num_shape:]], y_tr, epochs=10, batch_size=2048)
fraud_focal_model.fit([X_tr.iloc[:, :num_shape], X_tr.iloc[:, num_shape:]], y_tr, epochs=10, batch_size=2048)


# In[49]:


test_preds = fraud_model.predict([X_test.iloc[:, :num_shape], X_test.iloc[:, num_shape:]], batch_size = 8000)
test_preds2 = fraud_focal_model.predict([X_test.iloc[:, :num_shape], X_test.iloc[:, num_shape:]], batch_size = 8000)


# In[50]:


sub['isFraud'] = rankdata(test_preds, method='dense') + rankdata(test_preds2, method='dense')
sub.isFraud = sub.isFraud/sub.isFraud.max()
sub.to_csv('submission.csv', index=False)

