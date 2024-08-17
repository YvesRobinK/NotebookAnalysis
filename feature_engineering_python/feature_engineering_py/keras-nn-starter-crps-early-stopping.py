#!/usr/bin/env python
# coding: utf-8

# # Keras Starter & Metric CRPS & Early Stopping 🚀
# 
# **Intro:**
# 1. Only have done some naive aggregations by PlayId
# 2. Fully Connected NN
# 3. Cum Sum the Softmax output and clip to 0,1
# 4. Early Stopping Support for CRPS and restoring back to the "best" weights
# 5. A Scalable codebase that allows you to edid
# 6. Please **Upvote** this kernel! 👍🏻

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from kaggle.competitions import nflrush

import io
import re
from pprint import pprint
import numpy as np
import pandas as pd
import tensorflow
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.engine.saving import load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import multiprocessing
from keras import backend as F


# In[ ]:





# In[2]:


env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# # Main Functions

# In[3]:


def generate_categorical_encoders(train, features):
    encoders = {}
    for feature in features:
        train[feature] = train[feature].fillna('missing')
        encoder = LabelEncoder()
        encoder.fit(train[feature].values)
        le_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        encoders[feature] = le_dict
    return encoders


def encode_categorical_features(df, features, encoders):
    for f in features:
        df[f] = df[f].fillna('missing')
        df[f] = df[f].map(encoders[f])

def aggreate_by_play(df, configs):
    df = df.sort_values('PlayId')
    agg_df = pd.DataFrame({
        'PlayId': list(df['PlayId'].unique())
    }).sort_values('PlayId')
    # TODO aggerate with a sliding window
    for config in configs:
        feature = config[0]
        if feature == 'PlayId' or feature not in df.columns:
            continue
        gy = df.groupby('PlayId')
        for agg_func in config[2]:
            if agg_func == 'first':
                agg_df[feature] = gy[feature].agg(agg_func).values
            else:
                agg_df[f'{feature}_{agg_func}'] = gy[feature].agg(agg_func).values
    return agg_df

class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train.shape[0])
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid.shape[0])
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


class Gambler():
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.models = {}
        self.scalers = {}

    def construct_model(self):
        opm = Adam(learning_rate=0.001)
        # opm = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model = Sequential([
            Dense(128, input_shape=(self.input_size,)),
            Activation('relu'),
            Dense(64),
            Activation('relu'),
            Dense(199),
            Activation('softmax'),
        ])

        model.compile(loss='categorical_crossentropy',
                      optimizer=opm,
                      metrics=[])
        return model

    def train(self, X_train, X_valid, y_train, y_valid, fold):
        self.models[fold] = self.construct_model()

        self.scalers[fold] = StandardScaler()
        X_train = self.scalers[fold].fit_transform(X_train)
        X_valid = self.scalers[fold].transform(X_valid)

        es = EarlyStopping(monitor='val_CRPS',
                           mode='min',
                           restore_best_weights=True,
                           verbose=2,
                           patience=5)
        es.set_model(self.models[fold])
        metric = Metric(self.models[fold], [es], [(X_train, y_train), (X_valid, y_valid)])
        self.models[fold].fit(X_train, y_train,
                              verbose=0,
                              callbacks=[metric],
                              epochs=1000, batch_size=128)

    def predict(self, X, fold):
        X = self.scalers[fold].transform(X)
        preds = self.models[fold].predict(X)
        return preds
    
    def predict_final(self, X):
        final = None
        for fold in self.models.keys():
            preds = self.predict(X, fold)
            if final is None:
                final = preds / (len(self.models.keys()))
            else:
                final += preds / (len(self.models.keys()))
        return final

def train_loop(gambler, df, num_folds):
    spliter = KFold(n_splits=num_folds)
    oof_predictions = np.zeros((df.shape[0], 199))
    oof_targets = np.zeros((df.shape[0], 199))
    oof_ids = np.zeros(df.shape[0])
    fold = 0
    for train_index, valid_index in spliter.split(df):
        print('###', fold, '###')
        dataset_train = df.loc[train_index].copy()
        dataset_valid = df.loc[valid_index].copy()

        X_train = dataset_train[useful_raw_features].copy().fillna(-10)
        X_valid = dataset_valid[useful_raw_features].copy().fillna(-10)

        # get targets
        targets = dataset_train['Yards']
        y_train = np.zeros((targets.shape[0], 199))
        for idx, target in enumerate(list(targets)):
            y_train[idx][99 + target] = 1

        targets = dataset_valid['Yards']
        y_valid = np.zeros((targets.shape[0], 199))
        for idx, target in enumerate(list(targets)):
            y_valid[idx][99 + target] = 1

        gambler.train(X_train, X_valid, y_train, y_valid, fold)
    
        oof_predictions[valid_index] = gambler.predict(X_valid, fold)
        oof_targets[valid_index] = y_valid
        oof_ids[valid_index] = dataset_valid['PlayId'].values
        fold += 1
    return oof_ids, oof_predictions, oof_targets


# # Aggreation Configs

# In[4]:


raw_feature_configs = [
    ('GameId', 2, ['first']),
    ('PlayId', 2, ['first']),
    ('Team', 2, ['first']),
    ('X', 0, ['max', 'mean', 'std']),
    ('Y', 0, ['max', 'mean', 'std']),
    ('S', 0, ['max', 'mean', 'std']),
    ('A', 0, ['max', 'mean', 'std']),
    ('Dis', 0, ['mean']),
    ('Orientation', 0, ['mean']),
    ('Dir', 0, ['mean']),
    # ('NflId', 1, ['expand']),
    # ('DisplayName', 2, []),
    # ('JerseyNumber', 2, []),
    ('Season', 1, ['first']),
    ('YardLine', 0, ['mean']),
    # ('Quarter', 1, ['first']),
    ('GameClock', 0, ['first']),
    ('PossessionTeam', 1, ['first']),
    ('Down', 2, ['first']),
    ('Distance', 0, ['first']),
    ('FieldPosition', 1, ['first']),
    ('HomeScoreBeforePlay', 2, ['first']),
    ('VisitorScoreBeforePlay', 2, ['first']),
    ('NflIdRusher', 1, ['first']),
    # ('OffenseFormation', 1, ['first']),
    # ('OffensePersonnel', 1, ['first']),
    ('DefendersInTheBox', 0, ['first']),
    # ('DefensePersonnel', 1, ['first']),
    ('PlayDirection', 2, ['first']),
    # ('TimeHandoff', 2, []),
    ('TimeSnap', 2, ['first']),
    # ('PlayerHeight', 0, ['mean', 'max', 'min', 'std']),
    ('PlayerWeight', 0, ['mean', 'max', 'min', 'std']),
    ('PlayerBirthDate', 2, []),
    ('PlayerCollegeName', 2, []),
    ('HomeTeamAbbr', 1, ['first']),
    ('VisitorTeamAbbr', 1, ['first']),
    ('Week', 0, ['first']),
    ('Stadium', 1, ['first']),
    ('Location', 1, ['first']),
    ('StadiumType', 2, ['first']),
    ('Turf', 1, ['first']),
    ('GameWeather', 1, ['first']),
    ('Temperature', 0, ['first']),
    ('Humidity', 0, ['first']),
    ('WindSpeed', 0, ['first']),
    ('WindDirection', 1, ['first']),
    ('Yards', 2, ['first']),
]


# # Feature Engineering

# In[5]:


train_df['GameClock'] = train_df['GameClock'].str.replace(':', '').astype(int)
train_df['Age'] = [2019 - int(v.split('/')[2]) for v in train_df['PlayerBirthDate'].values]
raw_feature_configs.append(('Age', 0, ['mean', 'min', 'max']))


# # Model Training + Validation

# In[6]:


play_df = aggreate_by_play(train_df, raw_feature_configs)

unuse_features = [f[0] for f in raw_feature_configs if f[1] == 2]
useful_raw_features = [f for f in play_df.columns if f not in unuse_features]
categorical_features = [f for f in play_df.columns if
                        str(play_df[f].dtype) == 'object' and f not in unuse_features]

encoders = generate_categorical_encoders(play_df, categorical_features)
encode_categorical_features(play_df, categorical_features, encoders)

gambler = Gambler(len(useful_raw_features))
oof_ids, oof_predictions, oof_targets = train_loop(gambler, play_df, 5)

oof_targets = np.clip(np.cumsum(oof_targets, axis=1), 0, 1)
oof_predictions = np.clip(np.cumsum(oof_predictions, axis=1), 0, 1)

oof_score = ((oof_predictions - oof_targets) ** 2) \
                .sum(axis=1).sum(axis=0) / (199 * oof_targets.shape[0])
print('out of fold score', oof_score)


# # Test Prediction & Probability Distribution Visualization

# In[7]:


from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

count = 0
with tqdm_notebook(total=3438) as pbar:
    for (test_df, sample_prediction_df) in (env.iter_test()):
        test_df['GameClock'] = test_df['GameClock'].str.replace(':', '').astype(int)
        test_df['Age'] = [2019 - int(v.split('/')[2]) for v in test_df['PlayerBirthDate'].values]
        
        play_df = aggreate_by_play(test_df, raw_feature_configs)
        encode_categorical_features(play_df, categorical_features, encoders)
        play_df = play_df[useful_raw_features].fillna(-10)

        # for visualization
        if count % 170 == 0:
            p_fold_0 = gambler.predict(play_df, 0)
            fp_fold_0 = np.clip(np.cumsum(p_fold_0, axis=1), 0, 1)

            p_fold_1 = gambler.predict(play_df, 1)
            fp_fold_1 = np.clip(np.cumsum(p_fold_1, axis=1), 0, 1)

            p_fold_2 = gambler.predict(play_df, 2)
            fp_fold_2 = np.clip(np.cumsum(p_fold_2, axis=1), 0, 1)

            p_fold_3 = gambler.predict(play_df, 3)
            fp_fold_3 = np.clip(np.cumsum(p_fold_3, axis=1), 0, 1)

            p_fold_4 = gambler.predict(play_df, 4)
            fp_fold_4 = np.clip(np.cumsum(p_fold_4, axis=1), 0, 1)
            
            p_fold_mean = gambler.predict_final(play_df)
            fp_fold_mean = np.clip(np.cumsum(p_fold_mean, axis=1), 0, 1)
            
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

            pd.Series(p_fold_0[0]).plot(ax=axes[0,0])
            pd.Series(p_fold_1[0]).plot(ax=axes[0,0])
            pd.Series(p_fold_2[0]).plot(ax=axes[0,0])
            pd.Series(p_fold_3[0]).plot(ax=axes[0,0])
            pd.Series(p_fold_4[0]).plot(ax=axes[0,0])
            pd.Series(p_fold_mean[0]).plot(ax=axes[0,1])
            pd.Series(fp_fold_mean[0]).plot(ax=axes[1,0])
            fig.suptitle(f'Prediction {count}, with PlayId {test_df["PlayId"].iloc[0]}')
            axes[0][0].set_title('Softmax outputs of 5 folds')
            axes[0][1].set_title('Mean of softmax outputs of 5 folds')
            axes[1][0].set_title('Cum. Sum of the mean of softmax outputs of 5 folds')
            plt.show()
            
            
        # prediction
        p = gambler.predict_final(play_df)
        p = np.clip(np.cumsum(p, axis=1), 0, 1)
        # submission
        submission_df = pd.DataFrame(data=p, columns=sample_prediction_df.columns)
        env.predict(submission_df)
        pbar.update(1)
        count += 1
        
env.write_submission_file()


# In[ ]:





# In[ ]:




