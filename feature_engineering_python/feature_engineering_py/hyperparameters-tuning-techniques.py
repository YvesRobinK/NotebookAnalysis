#!/usr/bin/env python
# coding: utf-8

# <h1><center>Titanic: hyperparameters tuning techniques</center></h1>
# 
# <center><img width="1000" height="800" src="https://www.dlt.travel/immagine/33923/magazine-titanic2.jpg"></center>

# ### Hello everyone! In this kernel I am going to present some most used techniques for hyperparameters optimization. Let's do it!
# 
# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:Black; border:0' role="tab" aria-controls="home"><center>Quick navigation</center></h3>
# 
# * [1. Feature engineering](#1)
# * [2. XGBoost with default parameters](#2)
# * [3. Grid Search hyperparameters optimization](#3)
# * [4. Optuna hyperparameters optimization](#4)
# * [5. Hyperopt](#5)
# * [6. Ray Tune](#6)
# * [7. Skopt](#7)
# * [8. TBD](#8)

# In[1]:


import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold

import optuna
from optuna.samplers import TPESampler

from hyperopt import STATUS_OK, fmin, hp, tpe
from ray import tune
from skopt import BayesSearchCV


# <a id="1"></a>
# <h2 style='background:black; border:0; color:white'><center>1. Feature engineering<center><h2>

# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


train['LastName'] = train['Name'].str.split(',', expand=True)[0]
test['LastName'] = test['Name'].str.split(',', expand=True)[0]
ds = pd.concat([train, test])

sur = list()
died = list()

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

for cat in ['Pclass', 'Sex', 'Embarked', 'Cabin']:
    train = pd.concat([train, pd.get_dummies(train[cat], prefix=cat)], axis=1)
    train = train.drop([cat], axis=1)
    test = pd.concat([test, pd.get_dummies(test[cat], prefix=cat)], axis=1)
    test = test.drop([cat], axis=1)
    
train = train.drop(['PassengerId', 'Ticket', 'LastName', 'SibSp', 'Parch', 'Sex_male', 'Name'], axis=1)
test =  test.drop(['PassengerId', 'Ticket', 'LastName', 'SibSp', 'Parch', 'Sex_male', 'Name'], axis=1)

train = train.fillna(-1)
test = test.fillna(-1)

train.head()


# In[4]:


y = train['Survived']
X = train.drop(['Survived', 'Cabin_T'], axis=1)
X_test = test.copy()

X, X_val, y, y_val = train_test_split(X, y, random_state=666, test_size=0.2, shuffle=False)


# <a id="2"></a>
# <h2 style='background:black; border:0; color:white'><center>2. XGBoost with default parameters<center><h2>

# ## LB score: 0.77272

# In[5]:


model = XGBClassifier(
    random_state=666
)
model.fit(X, y)
preds = model.predict(X_val)

print('Default XGB accuracy: ', accuracy_score(y_val, preds))
print('Default XGB f1-score: ', f1_score(y_val, preds))


# In[6]:


preds = model.predict(X_test)
preds = preds.astype(np.int16)

submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('default_submission.csv', index=False)


# <a id="3"></a>
# <h2 style='background:black; border:0; color:white'><center>3. Grid Search hyperparameters optimization<center><h2>

# ## LB score: 0.78708

# In[7]:


get_ipython().run_cell_magic('time', '', "\nparameters = {\n    'max_depth': [4, 5, 6],\n    'n_estimators': [70, 80, 90, 100, 120, 150],\n    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.35, 0.5], \n    'gamma': [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]\n}\n\nestimator = XGBClassifier(\n    random_state=666\n)\n\nclf = GridSearchCV(\n    estimator, \n    parameters\n)\n\nclf.fit(X, y)\n")


# In[8]:


grid_search_params = clf.best_params_
grid_search_params['random_state'] = 666
grid_search_params


# In[9]:


grid_xgb = XGBClassifier(
    **grid_search_params
)

grid_xgb.fit(X, y)
preds = grid_xgb.predict(X_val)

print('Grid Search XGB accuracy: ', accuracy_score(y_val, preds))
print('Grid Search XGB f1-score: ', f1_score(y_val, preds))


# In[10]:


preds = grid_xgb.predict(X_test)
preds = preds.astype(np.int16)

submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('grid_search_submission.csv', index=False)


# <a id="4"></a>
# <h2 style='background:black; border:0; color:white'><center>4. Optuna hyperparameters optimization<center><h2>

# ## LB score: 0.77990

# In[11]:


# To see optuna progress you need to comment this row
optuna.logging.set_verbosity(optuna.logging.WARNING)


# In[12]:


class Optimizer:
    def __init__(self, metric, trials=50):
        self.metric = metric
        self.trials = trials
        self.sampler = TPESampler(seed=666)
        
    def objective(self, trial):
        model = create_model(trial)
        model.fit(X, y)
        preds = model.predict(X_val)
        if self.metric == 'acc':
            return accuracy_score(y_val, preds)
        else:
            return f1_score(y_val, preds)
            
    def optimize(self):
        study = optuna.create_study(
            direction="maximize", 
            sampler=self.sampler
        )
        study.optimize(
            self.objective, 
            n_trials=self.trials
        )
        return study.best_params


# In[13]:


get_ipython().run_cell_magic('time', '', '\ndef create_model(trial):\n    max_depth = trial.suggest_int("max_depth", 2, 6)\n    n_estimators = trial.suggest_int("n_estimators", 1, 150)\n    learning_rate = trial.suggest_uniform(\'learning_rate\', 0.0000001, 1)\n    gamma = trial.suggest_uniform(\'gamma\', 0.0000001, 1)\n    model = XGBClassifier(\n        learning_rate=learning_rate, \n        n_estimators=n_estimators, \n        max_depth=max_depth, \n        gamma=gamma, \n        random_state=666\n    )\n    return model\n\noptimizer = Optimizer(\'acc\', 100)\noptuna_params = optimizer.optimize()\noptuna_params[\'random_state\'] = 666\noptuna_params\n')


# In[14]:


optuna_xgb = XGBClassifier(
    **optuna_params
)
optuna_xgb.fit(X, y)
preds = optuna_xgb.predict(X_val)

print('Optuna XGB accuracy: ', accuracy_score(y_val, preds))
print('Optuna XGB f1-score: ', f1_score(y_val, preds))


# In[15]:


preds = optuna_xgb.predict(X_test)
preds = preds.astype(np.int16)

submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('optuna_submission.csv', index=False)


# <a id="5"></a>
# <h2 style='background:black; border:0; color:white'><center>5. Hyperopt<center><h2>

# ## LB score: 0.77990

# In[16]:


def score(params):
    model = XGBClassifier(
        **params
    )
    model.fit(X, y)
    predictions = model.predict(X_val)
    
    return {
        'loss': 1 - accuracy_score(y_val, predictions), 
        'status': STATUS_OK
    }


# In[17]:


get_ipython().run_cell_magic('time', '', "\nspace = {\n    'n_estimators': hp.choice('n_estimators', range(1, 150, 1)),\n    'learning_rate': hp.quniform('learning_rate', 0.0005, 1, 0.0005),\n    'max_depth':  hp.choice('max_depth', range(2, 6, 1)),\n    'gamma': hp.quniform('gamma', 0.0005, 1, 0.0025),\n    'random_state': 666\n}\n\nbest = fmin(\n    score, \n    space, \n    algo=tpe.suggest, \n    max_evals=1000\n)\n\nbest\n")


# In[18]:


hyperopt_xgb = XGBClassifier(
    **best
)
hyperopt_xgb.fit(X, y)
preds = hyperopt_xgb.predict(X_val)

print('Hyperopt XGB accuracy: ', accuracy_score(y_val, preds))
print('Hyperopt XGB f1-score: ', f1_score(y_val, preds))


# In[19]:


preds = hyperopt_xgb.predict(X_test)
preds = preds.astype(np.int16)

submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('hyperopt_submission.csv', index=False)


# <a id="6"></a>
# <h2 style='background:black; border:0; color:white'><center>6. Ray Tune<center><h2>

# ## LB score: 0.77990 (version 12)

# In[20]:


config = {
    "max_depth": tune.randint(2, 6),
    "n_estimators": tune.randint(1, 150),
    "gamma": tune.uniform(0.0, 1.0),
    "learning_rate": tune.uniform(0.0, 1.0),
    'random_state': 666
}


# In[21]:


def train_ray(config):
    model = XGBClassifier(
        **config
    )
    
    model.fit(X, y)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    
    tune.report(
        mean_accuracy=accuracy, 
        done=True
    )


# In[22]:


get_ipython().run_cell_magic('time', '', '\nanalysis = tune.run(\n    train_ray,\n    metric="mean_accuracy",\n    mode="max",\n    config=config,\n    num_samples=300,\n    verbose=-1\n)\n')


# In[23]:


ray_params = analysis.best_config
ray_params


# In[24]:


ray_xgb = XGBClassifier(
    **ray_params
)

ray_xgb.fit(X, y)
preds = ray_xgb.predict(X_val)

print('Ray Tune XGB accuracy: ', accuracy_score(y_val, preds))
print('Ray Tune XGB f1-score: ', f1_score(y_val, preds))


# In[25]:


preds = ray_xgb.predict(X_test)
preds = preds.astype(np.int16)

submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('raytune_submission.csv', index=False)


# <a id="7"></a>
# <h2 style='background:black; border:0; color:white'><center>7. Skopt<center><h2>

# ## LB score: 0.77990 (version 12)

# In[26]:


params = dict()
params['learning_rate'] = (0.000001, 1.0, 'log-uniform')
params['gamma'] = (0.000001, 1.0, 'log-uniform')
params['max_depth'] = (2, 6)
params['n_estimators'] = (1, 150)


# In[27]:


get_ipython().run_cell_magic('time', '', '\ncv = RepeatedStratifiedKFold(\n    n_splits=8, \n    n_repeats=5, \n    random_state=666\n)\n\nsearch = BayesSearchCV(\n    estimator=XGBClassifier(\n        random_state=666\n    ), \n    search_spaces=params, \n    cv=cv\n)\n\nsearch.fit(X, y)\nsearch.best_params_\n')


# In[28]:


skopt_params = dict()

for param in search.best_params_:
    skopt_params[param] = search.best_params_[param]

skopt_params['random_state'] = 666


# In[29]:


skopt_xgb = XGBClassifier(
    **skopt_params
)

skopt_xgb.fit(X, y)
preds = skopt_xgb.predict(X_val)

print('Ray Tune XGB accuracy: ', accuracy_score(y_val, preds))
print('Ray Tune XGB f1-score: ', f1_score(y_val, preds))


# In[30]:


preds = skopt_xgb.predict(X_test)
preds = preds.astype(np.int16)

submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = preds
submission.to_csv('skopt_submission.csv', index=False)


# <a id="8"></a>
# <h2 style='background:black; border:0; color:white'><center>8. WORK IN PROGRESS<center><h2>

# In[ ]:




