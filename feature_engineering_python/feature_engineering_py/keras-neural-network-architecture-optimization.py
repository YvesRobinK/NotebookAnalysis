#!/usr/bin/env python
# coding: utf-8

# <h1><center>Titanic: Keras Neural Network architecture optimization</center></h1>
# 
# <center><img src="https://www.dlt.travel/immagine/33923/magazine-titanic2.jpg"></center>

# #### In this kernel I present a simple approach to optimize keras neural network architecture using optuna. Method is similar to one presented here: <a href="https://www.kaggle.com/isaienkov/top-10-efficient-ensembling-in-few-lines-of-code">Top 10%. Efficient ensembling in few lines of code</a>

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


train.head()


# First let's do some feature engineering.

# In[4]:


train['LastName'] = train['Name'].str.split(',', expand=True)[0]
test['LastName'] = test['Name'].str.split(',', expand=True)[0]
ds = pd.concat([train, test])

sur = []
died = []
for index, row in ds.iterrows():
    s = ds[(ds['LastName']==row['LastName']) & (ds['Survived']==1)]
    d = ds[(ds['LastName']==row['LastName']) & (ds['Survived']==0)]
    s=len(s)
    if row['Survived'] == 1:
        s-=1
    d=len(d)
    if row['Survived'] == 0:
        d-=1
    sur.append(s)
    died.append(d)
ds['FamilySurvived'] = sur
ds['FamilyDied'] = died

ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
ds['IsAlone'] = 0
ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
ds['Fare'] = ds['Fare'].fillna(train['Fare'].median())
ds['Embarked'] = ds['Embarked'].fillna('Q')

train = ds[ds['Survived'].notnull()]
test = ds[ds['Survived'].isnull()]
test = test.drop(['Survived'], axis=1)

train['rich_woman'] = 0
test['rich_woman'] = 0
train['men_3'] = 0
test['men_3'] = 0

train.loc[(train['Pclass']<=2) & (train['Sex']=='female'), 'rich_woman'] = 1
test.loc[(test['Pclass']<=2) & (test['Sex']=='female'), 'rich_woman'] = 1
train.loc[(train['Pclass']==3) & (train['Sex']=='male'), 'men_3'] = 1
test.loc[(test['Pclass']==3) & (test['Sex']=='male'), 'men_3'] = 1

train['rich_woman'] = train['rich_woman'].astype(np.int8)
test['rich_woman'] = test['rich_woman'].astype(np.int8)

train["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin']])
test['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test['Cabin']])

train = train.drop(['PassengerId', 'Ticket', 'LastName', 'SibSp', 'Parch'], axis=1)
test = test.drop(['PassengerId', 'Ticket', 'LastName', 'SibSp', 'Parch'], axis=1)

categorical = ['Pclass', 'Sex', 'Embarked', 'Cabin']
for cat in categorical:
    train = pd.concat([train, pd.get_dummies(train[cat], prefix=cat)], axis=1)
    train = train.drop([cat], axis=1)
    test = pd.concat([test, pd.get_dummies(test[cat], prefix=cat)], axis=1)
    test = test.drop([cat], axis=1)
    
train = train.drop(['Sex_male', 'Name'], axis=1)
test =  test.drop(['Sex_male', 'Name'], axis=1)

train = train.fillna(-1)
test = test.fillna(-1)
train.head()


# Here we specify default keras parameters for initial model

# In[5]:


EPOCHS = 15

initial_keras_params = {
    'layers_number': 1,
    'n_units_l_0': 128,
    'activation_l_0': 'relu',
    'dropout_l_0': 0.5,
    'lr': 0.001
}


# Now we build our keras classifier that will be used for optimization

# In[6]:


def keras_classifier(parameters):
    
    model = Sequential()
    layers_number = int(parameters['layers_number'])
    
    for i in range(layers_number):
        model.add(Dense(int(parameters['n_units_l_' + str(i)]), activation=parameters['activation_l_' + str(i)]))
        model.add(Dropout(int(parameters['dropout_l_' + str(i)])))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(lr=float(parameters['lr'])), 
        metrics=['accuracy']
    )
    return model


# Let's check our initial model

# In[7]:


model = keras_classifier(initial_keras_params)


# In[8]:


y = train['Survived']
y = tf.keras.utils.to_categorical(y, num_classes=2, dtype='float32')
X = train.drop(['Survived', 'Cabin_T'], axis=1)
X_test = test.copy()

X, X_val, y, y_val = train_test_split(X, y, random_state=0, test_size=0.2, shuffle=False)


# In[9]:


model.fit(X, y, validation_split=0.2, epochs=EPOCHS, batch_size=32)


# In[10]:


preds = model.predict(X_val)
preds = np.argmax(preds, axis=1)

print('accuracy: ', accuracy_score(np.argmax(y_val, axis=1), preds))
print('f1-score: ', f1_score(np.argmax(y_val, axis=1), preds))


# Let's start optimization process

# In[11]:


def create_model(trial):
    n_layers = trial.suggest_int("layers_number", 1, 2)
    model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l_{}".format(i), 2, 16)
        activation = trial.suggest_categorical('activation_l_{}'.format(i), ['relu', 'sigmoid', 'tanh', 'elu'])
        model.add(Dense(num_hidden, activation=activation))
        dropout = trial.suggest_uniform("dropout_l_{}".format(i), 0.1, 0.4)
        model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))

    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=['accuracy']
    )

    return model


# In[12]:


def objective(trial):
    model = create_model(trial)
    
    epochs = trial.suggest_int("epochs", 3, 20)
    batch = trial.suggest_int("batch", 1, X.shape[0] / 4)
    
    model.fit(
        X, 
        y, 
        batch_size=batch, 
        epochs=epochs, 
        verbose=0
    )
    preds = model.predict(X_val)
    return accuracy_score(np.argmax(y_val, axis=1), np.argmax(preds, axis=1))


# In[13]:


def optimize():
    sampler = TPESampler(seed=666)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=80)
    return study.best_params


# In[14]:


params = optimize()


# Our best parameters

# In[15]:


params


# In[16]:


epochs = params['epochs']
batch = params['batch']
del params['epochs']
del params['batch']

opt_model = keras_classifier(params)
opt_model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=batch)


# In[17]:


preds = opt_model.predict(X_val)
preds = np.argmax(preds, axis=1)
print('accuracy: ', accuracy_score(np.argmax(y_val, axis=1), preds))
print('f1-score: ', f1_score(np.argmax(y_val, axis=1), preds))


# In[18]:


preds = opt_model.predict(X_test)
preds = np.argmax(preds, axis=1)
preds = preds.astype(np.int16)


# In[19]:


submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('submission.csv', index=False)


# In[20]:


submission.head()


# In[ ]:




