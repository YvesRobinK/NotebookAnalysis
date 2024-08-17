#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import datetime as dt
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn import impute
from sklearn import preprocessing
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 7)


# In[2]:


PATH = Path('../input/football-match-probability-prediction')
get_ipython().system('ls {PATH}')


# # Code forked from https://www.kaggle.com/igorkf
# 
# https://www.kaggle.com/igorkf/football-match-probability-prediction-lstm-starter

# In[3]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv(PATH / 'train.csv')\ntest = pd.read_csv(PATH / 'test.csv')\n")


# In[4]:


for col in train.filter(regex='date', axis=1).columns:
    train[col] = pd.to_datetime(train[col])
    test[col] = pd.to_datetime(test[col])
    
# date based features
for i in range(1, 11):
    train[f'home_team_history_match_days_ago_{i}'] = (train['match_date'] - train[f'home_team_history_match_date_{i}']).dt.days
    train[f'away_team_history_match_days_ago_{i}'] = (train['match_date'] - train[f'away_team_history_match_date_{i}']).dt.days
    test[f'home_team_history_match_days_ago_{i}'] = (test['match_date'] - test[f'home_team_history_match_date_{i}']).dt.days
    test[f'away_team_history_match_days_ago_{i}'] = (test['match_date'] - test[f'away_team_history_match_date_{i}']).dt.days
    
# remove two matchs with possible error
combined_train=train
combined_train['tag']='Train_Data'

combined_test=test
combined_test['tag']='Test_Data'

combined=combined_train.append(combined_test)
combined['year']=pd.to_datetime(combined['match_date']).dt.year

import datetime as dt
rating_features=[]
combined['date_difference_last_match_home']=(pd.to_datetime(combined['match_date'])-pd.to_datetime(combined['home_team_history_match_date_1'])).dt.days
combined['date_difference_last_match_away']=(pd.to_datetime(combined['match_date'])-pd.to_datetime(combined['away_team_history_match_date_1'])).dt.days

combined['Goals_For_Home']=combined['home_team_history_goal_1']+combined['home_team_history_goal_2']+combined['home_team_history_goal_3']+combined['home_team_history_goal_4']+combined['home_team_history_goal_5']
combined['Opponent_Goals_For_Home']=combined['home_team_history_opponent_goal_1']+combined['home_team_history_opponent_goal_2']+combined['home_team_history_opponent_goal_3']+combined['home_team_history_opponent_goal_4']+combined['home_team_history_opponent_goal_5']
combined['goal_difference']=combined['Goals_For_Home']-combined['Opponent_Goals_For_Home']

combined['Goals_For_Away']=combined['away_team_history_goal_1']+combined['away_team_history_goal_2']+combined['away_team_history_goal_3']+combined['away_team_history_goal_4']+combined['away_team_history_goal_5']
combined['Opponent_Goals_For_Away']=combined['away_team_history_opponent_goal_1']+combined['away_team_history_opponent_goal_2']+combined['away_team_history_opponent_goal_3']+combined['away_team_history_opponent_goal_4']+combined['away_team_history_opponent_goal_5']
combined['goal_difference_away']=combined['Goals_For_Away']-combined['Opponent_Goals_For_Away']

#Performance in Last 5 matches
for i in range(1,11):
    combined['last_match_result_home_'+str(i)]=np.where(combined['home_team_history_goal_'+str(i)]>combined['home_team_history_opponent_goal_'+str(i)],3,np.where(combined['home_team_history_goal_'+str(i)]<combined['home_team_history_opponent_goal_'+str(i)],0,1))
    rating_features.append('last_match_result_home_'+str(i))
    
    combined['last_match_result_away_'+str(i)]=np.where(combined['away_team_history_goal_'+str(i)]>combined['away_team_history_opponent_goal_'+str(i)],0,np.where(combined['away_team_history_goal_'+str(i)]<combined['away_team_history_opponent_goal_'+str(i)],3,1))
    rating_features.append('last_match_result_away_'+str(i))

combined['home_performance_last_5']=combined['last_match_result_home_1']+combined['last_match_result_home_2']+combined['last_match_result_home_3']+combined['last_match_result_home_4']+combined['last_match_result_home_5']
combined['away_performance_last_5']=combined['last_match_result_away_1']+combined['last_match_result_away_2']+combined['last_match_result_away_3']+combined['last_match_result_away_4']+combined['last_match_result_away_5']
combined['home_vs_away']=combined['home_performance_last_5']+combined['away_performance_last_5']


rating_features=rating_features+['home_performance_last_5','away_performance_last_5','home_vs_away']

def streak(m1,m2,m3,m4,m5,k):
    streak=0
    
    lst_results=[m1,m2,m3,m4,m5]
    for i in lst_results:
        if(i==k):
            streak=streak+1
        else:
            return streak
            break
    
    return streak


combined['home_same_coach']=combined.apply(lambda x:streak(x.home_team_history_coach_1,x.home_team_history_coach_2,x.home_team_history_coach_3,x.home_team_history_coach_4,x.home_team_history_coach_5,x.home_team_coach_id),axis=1)
combined['away_same_coach']=combined.apply(lambda x:streak(x.away_team_history_coach_1,x.away_team_history_coach_2,x.away_team_history_coach_3,x.away_team_history_coach_4,x.away_team_history_coach_5,x.away_team_coach_id),axis=1)



mapping_freq=40

frequencies=combined['home_team_name'].value_counts()
mapping=combined['home_team_name'].map(frequencies)
combined['home_team_name_mapping']=combined['home_team_name'].mask(mapping<mapping_freq,'Other')

frequencies=combined['away_team_name'].value_counts()
mapping=combined['away_team_name'].map(frequencies)
combined['away_team_name_mapping']=combined['away_team_name'].mask(mapping<mapping_freq,'Other')

combined['away_coach_change']=np.where(combined['away_team_coach_id']==combined['away_team_history_coach_1'],0,1)


combined['home_team_coach_id']=combined['home_team_coach_id'].astype('str')
frequencies=combined['home_team_name'].value_counts()
mapping=combined['home_team_name'].map(frequencies)
combined['home_team_name_mapping']=combined['home_team_name'].mask(mapping<mapping_freq,'Other')


# Coach Mapping
combined['home_team_coach_id']=combined['home_team_coach_id'].astype('str')
frequencies=combined['home_team_coach_id'].value_counts()
mapping=combined['home_team_coach_id'].map(frequencies)
combined['home_team_coach_mapping']=combined['home_team_coach_id'].mask(mapping<mapping_freq,'Other')

combined['away_team_coach_id']=combined['away_team_coach_id'].astype('str')
frequencies=combined['away_team_coach_id'].value_counts()
mapping=combined['away_team_coach_id'].map(frequencies)
combined['away_team_coach_mapping']=combined['away_team_coach_id'].mask(mapping<mapping_freq,'Other')




combined['home_winning_streak']=combined.apply(lambda x:streak(x.last_match_result_home_1,x.last_match_result_home_2,x.last_match_result_home_3,x.last_match_result_home_4,x.last_match_result_home_5,3),axis=1)
combined['home_losing_streak']=combined.apply(lambda x:streak(x.last_match_result_home_1,x.last_match_result_home_2,x.last_match_result_home_3,x.last_match_result_home_4,x.last_match_result_home_5,0),axis=1)
combined['home_draw_streak']=combined.apply(lambda x:streak(x.last_match_result_home_1,x.last_match_result_home_2,x.last_match_result_home_3,x.last_match_result_home_4,x.last_match_result_home_5,1),axis=1)

combined['away_winning_streak']=combined.apply(lambda x:streak(x.last_match_result_away_1,x.last_match_result_away_2,x.last_match_result_away_3,x.last_match_result_away_4,x.last_match_result_away_5,3),axis=1)
combined['away_losing_streak']=combined.apply(lambda x:streak(x.last_match_result_away_1,x.last_match_result_away_2,x.last_match_result_away_3,x.last_match_result_away_4,x.last_match_result_away_5,0),axis=1)
combined['away_draw_streak']=combined.apply(lambda x:streak(x.last_match_result_away_1,x.last_match_result_away_2,x.last_match_result_away_3,x.last_match_result_away_4,x.last_match_result_away_5,1),axis=1)
# combined=pd.get_dummies(combined)
train=combined[combined['tag']=='Train_Data']
test=combined[combined['tag']=='Test_Data']
combined


# In[5]:


rating_features=['date_difference_last_match_home',
       'date_difference_last_match_away', 'Goals_For_Home',
       'Opponent_Goals_For_Home', 'goal_difference', 'Goals_For_Away',
       'Opponent_Goals_For_Away', 'goal_difference_away',
       'last_match_result_home_1', 'last_match_result_away_1',
       'last_match_result_home_2', 'last_match_result_away_2',
       'last_match_result_home_3', 'last_match_result_away_3',
       'last_match_result_home_4', 'last_match_result_away_4',
       'last_match_result_home_5', 'last_match_result_away_5',
       'home_performance_last_5', 'away_performance_last_5', 'home_vs_away',
       'home_same_coach', 'away_same_coach', 'away_coach_change',
       
       'home_winning_streak', 'home_losing_streak', 'home_draw_streak',
       'away_winning_streak', 'away_losing_streak', 'away_draw_streak']


# In[6]:


combined


# ### Choosing a validation method

# In[7]:


print('Train:')
print('First match at:', train['match_date'].min())
print('Last match at: ', train['match_date'].max())
print('Range:', train['match_date'].max() - train['match_date'].min())


# In[8]:


print('Test:')
print('First match at:', test['match_date'].min())
print('Last match at: ', test['match_date'].max())
print('Range:', test['match_date'].max() - test['match_date'].min())


# Let's try to take 213 days to use on validation set

# In[9]:


validation_split = dt.datetime.strptime(
    '2021-05-01 00:00:00', '%Y-%m-%d %H:%M:%S') - dt.timedelta(days=50, hours=23, minutes=15)
print('Train until:', validation_split)


# ## Training a LSTM model

# In[10]:


class CFG:
    seed = 42
    classes = 3
    use_class_weights = False
    batch_size = 512
    epochs = 100
    early_stopping_patience = epochs // 10
    mask = -999.0
    timesteps = 10
    target = 'target_int'
    historical_features = [
        # home based
        'home_team_history_goal',
        'home_team_history_opponent_goal',
        'home_team_history_is_play_home', 
        'home_team_history_rating',
        'home_team_history_opponent_rating',
        'home_team_history_match_days_ago',
        
        # away based
        'away_team_history_goal', 
        'away_team_history_opponent_goal',
        'away_team_history_rating',
        'away_team_history_opponent_rating',
        'away_team_history_match_days_ago',

    ] 
    
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def wide_to_long(df, feature: str, timesteps: int, mask=CFG.mask):
    df_ = df.copy()
    if feature not in CFG.historical_features:
        features = [feature]
    else:
        features = [f'{feature}_{i}' for i in range(1, timesteps + 1)][::-1]
#     print(features)
    df_ = df_[['id'] + features]
    df_ = df_.fillna(mask)
    series = df_.set_index('id').stack().reset_index(level=1)[0].rename(feature)
    return series


# In[11]:


# maps
target2int = {'away': 0, 'draw': 1, 'home': 2}
int2target = {x[1]: x[0] for x in target2int.items()}

# encode target
train['target_int'] = train['target'].map(target2int)

X_test = test.copy()

# split train/val
X_train = train[train['match_date'] <= validation_split].reset_index(drop=True)
X_val = train[train['match_date'] > validation_split].reset_index(drop=True)

# preprocess
features_pattern = '_[0-9]|'.join(CFG.historical_features) + '_[0-9]'
features_to_preprocess = train.filter(regex=features_pattern, axis=1).columns.tolist()

features_to_preprocess=features_to_preprocess+rating_features
    
print('Features to preprocess:', len(features_to_preprocess))
# this Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range)
scaler = preprocessing.RobustScaler()
X_train_pre = pd.DataFrame(scaler.fit_transform(X_train[features_to_preprocess]), columns=features_to_preprocess)
X_train_pre=pd.concat([X_train[['home_team_coach_mapping','away_team_coach_mapping','home_team_name_mapping','away_team_name_mapping']], X_train_pre], axis=1)
X_train_pre=pd.get_dummies(X_train_pre)

X_train = pd.concat([X_train[['id', 'league_name', 'target_int','home_team_coach_mapping']], X_train_pre], axis=1)

X_val_pre = pd.DataFrame(scaler.transform(X_val[features_to_preprocess]), columns=features_to_preprocess)
X_val_pre=pd.concat([X_val[['home_team_coach_mapping','away_team_coach_mapping','home_team_name_mapping','away_team_name_mapping']], X_val_pre], axis=1)
X_val_pre=pd.get_dummies(X_val_pre)

X_val = pd.concat([X_val[['id', 'league_name', 'target_int']], X_val_pre], axis=1)
X_train,X_val = X_train.align(X_val, join='outer', axis=1, fill_value=0)

X_test_pre = pd.DataFrame(scaler.transform(X_test[features_to_preprocess]), columns=features_to_preprocess)
X_test_pre=pd.concat([X_test[['home_team_coach_mapping','away_team_coach_mapping','home_team_name_mapping','away_team_name_mapping']], X_test_pre], axis=1)
X_test_pre=pd.get_dummies(X_test_pre)

X_test = pd.concat([X_test[['id', 'league_name']], X_test_pre], axis=1)
X_train,X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)


# create targets
y_train = wide_to_long(X_train, 'target_int', timesteps=CFG.timesteps).values.reshape(-1, 1)
y_val = wide_to_long(X_val, 'target_int', timesteps=CFG.timesteps).values.reshape(-1, 1)

# create historical features
X_train = pd.concat([
    wide_to_long(X_train, feature=feature, timesteps=CFG.timesteps) for feature in CFG.historical_features
], axis=1).values.reshape(-1, CFG.timesteps, len(CFG.historical_features))
X_val = pd.concat([
    wide_to_long(X_val, feature=feature, timesteps=CFG.timesteps) for feature in CFG.historical_features
], axis=1).values.reshape(-1, CFG.timesteps, len(CFG.historical_features))
X_test = pd.concat([
    wide_to_long(X_test, feature=feature, timesteps=CFG.timesteps) for feature in CFG.historical_features
], axis=1).values.reshape(-1, CFG.timesteps, len(CFG.historical_features))

print(X_train.shape)
print(y_train.shape)
print()
print(X_val.shape)
print(y_val.shape)
print()
print(X_test.shape)


# In[12]:


# distribution of away, draw, home
print(np.bincount(y_train.ravel()) / len(y_train))
print(np.bincount(y_val.ravel()) / len(y_val))


# The `draw` class is less frequent than the others.  

# In[13]:


if CFG.use_class_weights:
    # https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train.ravel()
    )
    class_weights = {i: x for i, x in enumerate(class_weights)}
else:
    class_weights = None
print('Class weights:', class_weights)


# In[14]:


def create_model():

    # input
    input_ = tf.keras.Input(shape=X_train.shape[1:])
    mask = tf.keras.layers.Masking(mask_value=CFG.mask, input_shape=(X_train.shape[1:]))(input_)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, activation='tanh'))(mask)
    x = tf.keras.layers.Dropout(0.5)(x)  
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True, activation='tanh'))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)

    # output
    output = tf.keras.layers.Dense(CFG.classes, activation='softmax')(x)
    
    model = tf.keras.Model(
        inputs=[input_],
        outputs=[output]
    )

    return model


# In[15]:


get_ipython().run_cell_magic('time', '', "\nset_seed(CFG.seed)\n\n# callbacks\nes = tf.keras.callbacks.EarlyStopping(\n    patience=CFG.early_stopping_patience,\n    restore_best_weights=True,\n    verbose=1\n)\nrlrop = tf.keras.callbacks.ReduceLROnPlateau(\n    factor=0.8,\n    patience=CFG.early_stopping_patience // 2,\n    verbose=1\n)\n\nmodel = create_model()\nmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')\n\n# fit\nh = model.fit(\n    X_train,\n    y_train,\n    validation_data=(X_val, y_val),\n    batch_size=CFG.batch_size,\n    epochs=CFG.epochs,\n    callbacks=[\n        es,\n        rlrop\n    ]\n)\n")


# In[16]:


model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    batch_size=512,
    epochs=10
)


# In[17]:


X_train.shape


# In[18]:


import tensorflow
tensorflow.config.experimental.list_physical_devices()


# In[19]:


X_train.shape


# In[20]:


plt.plot(h.history['loss'], label='Train')
plt.plot(h.history['val_loss'], label='Val')
plt.legend();


# ### Evaluate

# In[21]:


def evaluate(model, X, y):
    probs = model.predict(X)
#     preds = np.argmax(probs, axis=1)
#     report = metrics.classification_report(y, preds)
#     print(report)
    df_sub = pd.DataFrame({
        'away': probs[:, 0],
        'draw': probs[:, 1],
        'home': probs[:, 2]
    })
    df_sub['draw']=np.where(df_sub['draw']>0.38,0.95,0)
    df_sub['away']=np.where(df_sub['draw']==0.95,0.025,df_sub['away'])
    df_sub['home']=np.where(df_sub['draw']==0.95,0.025,df_sub['home'])
    probs=df_sub.values
    logloss = metrics.log_loss(y, probs)
    print('Log loss:', logloss)
#     cm = metrics.confusion_matrix(y, preds)
#     disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(target2int.keys()))
#     disp.plot(cmap=plt.cm.Blues)
#     plt.show();


# In[22]:


# evaluate(model, X_val, y_val)


# In[23]:


# # probs = model.predict(X_val)
# df_sub = pd.DataFrame({
#     'away': probs[:, 0],
#     'draw': probs[:, 1],
#     'home': probs[:, 2]
# })
# df_sub['draw']=np.where(df_sub['draw']>0.35,1,0)
# df_sub['away']=np.where(df_sub['draw']==1,0,df_sub['away'])
# df_sub['home']=np.where(df_sub['draw']==1,0,df_sub['home'])


# Seems that the `draw` class is hurting the log loss.   

# ## Predict on test set

# In[24]:


# away, draw, home
probs_test = model.predict(X_test)
probs_test


# In[25]:


probs_test


# In[26]:


# pd.concat([
#     train['target_int'].value_counts(normalize=True).sort_index().rename('observed'),
#     pd.Series(np.argmax(df_sub.iloc[:, 1:].values, axis=1), name='predicted').value_counts(normalize=True).sort_index()
# ], axis=1)


# In[27]:


df_sub = pd.DataFrame({
    'id': test['id'],
    'away': probs_test[:, 0],
    'draw': probs_test[:, 1],
    'home': probs_test[:, 2]
})
df_sub


# Very few predicted draws compared to the observed target.   

# In[28]:


df_sub.iloc[:, 1:].plot.hist(alpha=0.5);


# In[29]:


# submission expects this order: id, home, away, draw
df_sub[['id', 'home', 'away', 'draw']].to_csv('submission.csv', index=False)


# In[30]:


get_ipython().system('head submission.csv')


# In[31]:


train


# In[ ]:




