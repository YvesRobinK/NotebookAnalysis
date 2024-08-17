#!/usr/bin/env python
# coding: utf-8

# 
# ## Importing data

# In[1]:


import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_ns_dtype

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
import gc
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

from metric import score # Import event detection ap score function

# These are variables to be used by the score function
column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}


# In[2]:


def reduce_mem_usage(df):
    
    """ 
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not is_datetime64_ns_dtype(df[col]) and not 'category':
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
                    df[col] = df[col].astype(np.int32)  
            else:
                df[col] = df[col].astype(np.float16)
        
    return df


# In[3]:


def feat_eng(df):
    
    df['series_id'] = df['series_id'].astype('category')
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df["timestamp"].dt.hour
    
    df.sort_values(['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    
    df['lids'] = np.maximum(0., df['enmo'] - 0.02)
    df['lids'] = df['lids'].rolling(f'{120*5}s', center=True, min_periods=1).agg('sum')
    df['lids'] = 100 / (df['lids'] + 1)
    df['lids'] = df['lids'].rolling(f'{360*5}s', center=True, min_periods=1).agg('mean').astype(np.float32)
    
    df["enmo"] = (df["enmo"]*1000).astype(np.int16)
    df["anglez"] = df["anglez"].astype(np.int16)
    df["anglezdiffabs"] = df["anglez"].diff().abs().astype(np.float32)
    
    for col in ['enmo', 'anglez', 'anglezdiffabs']:
        
        # periods in seconds        
        periods = [60,360,720,3600] 
        
        for n in periods:
            
            rol_args = {'window':f'{n+5}s', 'min_periods':10, 'center':True}
            
            for agg in ['median', 'mean', 'max', 'min', 'var']:
                df[f'{col}_{agg}_{n}'] = df[col].rolling(**rol_args).agg(agg).astype(np.float32).values
                gc.collect()
            
            if n == max(periods):
                df[f'{col}_mad_{n}'] = (df[col] - df[f'{col}_median_{n}']).abs().rolling(**rol_args).median().astype(np.float32)
            
            df[f'{col}_amplit_{n}'] = df[f'{col}_max_{n}']-df[f'{col}_min_{n}']
            df[f'{col}_amplit_{n}_min'] = df[f'{col}_amplit_{n}'].rolling(**rol_args).min().astype(np.float32).values
            
#             if col in ['enmo', 'anglez']:
            df[f'{col}_diff_{n}_max'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).max().astype(np.float32)
            df[f'{col}_diff_{n}_mean'] = df[f'{col}_max_{n}'].diff().abs().rolling(**rol_args).mean().astype(np.float32)

    
            gc.collect()
    
    df.reset_index(inplace=True)
    df.dropna(inplace=True)

    return df


# In[4]:


file = '/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train.parquet'

def feat_eng_by_id(idx):
    
    from warnings import simplefilter 
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    
    df  = pd.read_parquet(file, filters=[('series_id','=',idx)])
    df['awake'] = df['awake'].astype(np.int8)
    df = feat_eng(df)
    
    return df


# ### Training and validating 

# In[5]:


DEV = False

series_id  = pd.read_parquet(file, columns=['series_id'])
series_id = series_id.series_id.unique()

print(len(series_id))

if DEV:
    series_id = series_id[::10]


# In[6]:


weird_series = ['31011ade7c0a', 'a596ad0b82aa']

series_id = [s for s in series_id if s not in weird_series]


# In[7]:


get_ipython().run_cell_magic('time', '', '\ntrain = Parallel(n_jobs=6)(delayed(feat_eng_by_id)(i) for i in series_id)\ntrain = pd.concat(train, ignore_index=True)\n')


# In[8]:


# REDUCE train data by half
train = train.iloc[::60]


# In[9]:


drop_cols = ['series_id', 'step', 'timestamp']

X, y = train.drop(columns=drop_cols+['awake']), train['awake']

gc.collect()


# In[10]:


if not DEV:
    del train
    gc.collect()


# In[11]:


class EnsembleAvgProba():
    
    def __init__(self, classifiers):
        
        self.classifiers = classifiers
    
    def fit(self,X,y):
        
        for classifier in self.classifiers:                
            classifier.fit(X, y)
            gc.collect()
     
    def predict_proba(self, X):
        
        probs = []
        
        for m in self.classifiers:
            probs.append(m.predict_proba(X))
        
        probabilities = np.stack(probs)
        p = np.mean(probabilities, axis=0)
        
        return p 
    
    def predict(self, X):
        
        probs = []
        
        for m in self.classifiers:
            probs.append(m.predict(X))
        
        probabilities = np.stack(probs)
        p = np.mean(probabilities, axis=0)
        
        return p.round()


# In[12]:


lgb_params1 = {    
    'boosting_type':'gbdt',
    'num_leaves':31,
    'max_depth':6,
    'learning_rate':0.03,
    'n_estimators':850,
    'subsample_for_bin':200000,
    'min_child_weight':0.001,
    'min_child_samples':20,
    'subsample':0.9,
#     'colsample_bytree':0.7,
    'reg_alpha':0.05,
    'reg_lambda':0.05,
             }
import xgboost as xgb


# Training classifier

model = EnsembleAvgProba(classifiers=[
                    lgb.LGBMClassifier(random_state=42, **lgb_params1),
                    GradientBoostingClassifier(n_estimators=100,max_depth=5,min_samples_leaf=300,random_state=42),
                    RandomForestClassifier(n_estimators=500, min_samples_leaf=300, random_state=42, n_jobs=-1),
                    xgb.XGBClassifier(n_estimators=520,objective="binary:logistic", learning_rate=0.02, max_depth=7, random_state=42)    ]
                )


# In[13]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X, y)\n')


# In[14]:


feats = []

feat_imp = model.classifiers[0].booster_.feature_importance(importance_type='gain')
feat_imp = pd.Series(model.classifiers[0].feature_importances_, index=X.columns).sort_values()
feats.append(feat_imp)

for m in model.classifiers[1:]:
    feat_imp = pd.Series(m.feature_importances_, index=X.columns).sort_values()
    feats.append(feat_imp)


# In[15]:


feat_imp = pd.Series(pd.concat(feats, axis=1).mean(axis=1), index=feats[0].index).sort_values()
print('Columns with poor contribution', feat_imp[feat_imp<0.001].index, sep='\n')
fig = px.bar(x=feat_imp, y=feat_imp.index, orientation='h')
fig.show()


# In[16]:


feat_imp.sort_values().head(10)


# In[17]:


# del X, y
gc.collect()


# In[18]:


def get_events(idx, classifier, file='test_series.parquet') :
    
    test  = pd.read_parquet(f'/kaggle/input/child-mind-institute-detect-sleep-states/{file}',
                    filters=[('series_id','=',idx)])
    test = feat_eng(test)
    X_test = test.drop(columns=drop_cols)
    test = test[drop_cols]

    preds, probs = classifier.predict(X_test), classifier.predict_proba(X_test)[:, 1]
    
    test['prediction'] = preds
    test['prediction'] = test['prediction'].rolling(360+1, center=True).median()
    test['probability'] = probs
    
    test = test[test['prediction']!=2]
    
    test.loc[test['prediction']==0, 'probability'] = 1-test.loc[test['prediction']==0, 'probability']
    test['score'] = test['probability'].rolling(60*12*5, center=True, min_periods=10).mean().bfill().ffill()

    
    test['pred_diff'] = test['prediction'].diff()
    
    test['event'] = test['pred_diff'].replace({1:'wakeup', -1:'onset', 0:np.nan})
    
    test_wakeup = test[test['event']=='wakeup'].groupby(test['timestamp'].dt.date).agg('first')
    test_onset = test[test['event']=='onset'].groupby(test['timestamp'].dt.date).agg('last')
    test = pd.concat([test_wakeup, test_onset], ignore_index=True).sort_values('timestamp')

    return test


# In[19]:


cols_sub = ['series_id','step','event','score']

series_id  = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet', columns=['series_id'])
series_id = series_id.series_id.unique()

tests = []

for idx in series_id: 

    test =  get_events(idx, model)
    tests.append(test[cols_sub])


# In[20]:


submission = pd.concat(tests, ignore_index=True).reset_index(names='row_id')
submission.to_csv('submission.csv', index=False)
submission


# In[21]:


# from keras.layers import LSTM,GRU,Dense,Dropout,Activation,Embedding,Flatten,Bidirectional,MaxPooling2D,Conv1D,SpatialDropout1D,MaxPooling1D
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# maxlen =37
# trunc_type='post'

# oov_tok = "<OOV>"
# vocab_size = 300


# In[22]:


# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super().__init__()
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)

#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)
# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super().__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions
# embed_dim = 32  # Embedding size for each token
# num_heads = 2  # Number of attention heads
# ff_dim = 32  # Hidden layer size in feed forward network inside transformer

# inputs = layers.Input(shape=(maxlen,))
# embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
# x = embedding_layer(inputs)
# transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# x = transformer_block(x)
# x=layers.Bidirectional(LSTM(64, return_sequences=True))(x)
# x = layers.GlobalAveragePooling1D()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Dense(20, activation="relu")(x)
# x = layers.Dropout(0.1)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)


# In[23]:


# model.summary()
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.RMSprop(1e-3),metrics=['accuracy'])


# In[24]:


# from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

# pocket = EarlyStopping(monitor='val_exact_matched_accuracy', min_delta=0.001,
#                        patience=10, verbose=1, mode='max', 
#                        restore_best_weights = True)
# reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
#                               mode='auto',verbose=1)
# history=model.fit(xtrain,ytrain,epochs=50,batch_size=256,validation_data=(xtest,ytest),callbacks=[pocket,reduce_lr])
# # history = pd.DataFrame(history.history)

