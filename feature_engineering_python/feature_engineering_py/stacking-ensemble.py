#!/usr/bin/env python
# coding: utf-8

# This notebook shows how to perform stacking ensemble (a.k.a. stacked generalization).
# 
# In [Ensemble-learning meta-classifier for stacking](https://www.kaggle.com/remekkinas/ensemble-learning-meta-classifier-for-stacking), @remekkinas shares how to do stacking ensemble using `MLExtend'`s `StackingCVClassifier`.
# 
# To demonstrate how stacking works, this notebook shows how to prepare the baseline model predictions using cross-validation (CV), then use them for level-2 stacking. It trains four classifiers, Random Forests, Extremely Randomized Trees, LightGBM, and CatBoost as level-1 base models. It also uses  CV predictions of two models, LightGBM with DAE features and supervised DAE trained from my previous notebook, [Supervised Emphasized Denoising AutoEncoder](https://www.kaggle.com/jeongyoonlee/supervised-emphasized-denoising-autoencoder) to show why keeping CV predictions for **every** model is important. :)
# 
# The contents of this notebook are as follows:
# 1. **Feature Engineering**: Same as in the [Supervised Emphasized Denoising AutoEncoder](https://www.kaggle.com/jeongyoonlee/supervised-emphasized-denoising-autoencoder) and [AutoEncoder + Pseudo Label + AutoLGB](https://www.kaggle.com/jeongyoonlee/autoencoder-pseudo-label-autolgb).
# 2. **Level-1 Base Model Training**: Training four base models, Random Forests, Extremely Randomized Trees, LightGBM, and CatBoost using the same 5-fold CV.
# 3. **Level-2 Stacking**: Training the LightGBM model with CV predictions of base models, original features, and DAE features. Performing feature selection and hyperparameter optimization using `Kaggler`'s `AutoLGB`.
# 
# This notebook is inspired and/or based on other Kagglers' notebooks as follows:
# * [TPS-APR21-EDA+MODEL](https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model) by @udbhavpangotra
# * [Ensemble-learning meta-classifier for stacking](https://www.kaggle.com/remekkinas/ensemble-learning-meta-classifier-for-stacking) by @remekkinas
# * [TPS Apr 2021 pseudo labeling/voting ensemble](https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble?scriptVersionId=60616606) by @hiro5299834
# 
# Thanks!

# # Part 1: Data Loading & Feature Engineering

# In[1]:


from catboost import CatBoostClassifier
from joblib import dump
import lightgbm as lgb
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings


# In[2]:


get_ipython().system('pip install kaggler')


# In[3]:


import kaggler
from kaggler.model import AutoLGB
from kaggler.preprocessing import LabelEncoder

print(f'Kaggler: {kaggler.__version__}')


# In[4]:


warnings.simplefilter('ignore')
pd.set_option('max_columns', 100)


# In[5]:


data_dir = Path('/kaggle/input/tabular-playground-series-apr-2021/')
trn_file = data_dir / 'train.csv'
tst_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'
pseudo_label_file = '../input/tps-apr-2021-pseudo-label-dae/REMEK-TPS04-FINAL005.csv'
dae_feature_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/dae.csv'
lgb_dae_predict_val_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/lgb_dae.val.txt'
lgb_dae_predict_tst_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/lgb_dae.tst.txt'
lgb_dae_te_predict_val_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/lgb_dae_te.val.txt'
lgb_dae_te_predict_tst_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/lgb_dae_te.tst.txt'
sdae_dae_predict_val_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/sdae_dae.val.txt'
sdae_dae_predict_tst_file = '/kaggle/input/tps-apr-2021-pseudo-label-dae/sdae_dae.tst.txt'

target_col = 'Survived'
id_col = 'PassengerId'

feature_name = 'dae'
algo_name = 'esb'
model_name = f'{algo_name}_{feature_name}'

feature_file = f'{feature_name}.csv'
predict_val_file = f'{model_name}.val.txt'
predict_tst_file = f'{model_name}.tst.txt'
submission_file = f'{model_name}.sub.csv'


# In[6]:


n_fold = 5
seed = 42
n_est = 1000
encoding_dim = 128


# In[7]:


trn = pd.read_csv(trn_file, index_col=id_col)
tst = pd.read_csv(tst_file, index_col=id_col)
sub = pd.read_csv(sample_file, index_col=id_col)
pseudo_label = pd.read_csv(pseudo_label_file, index_col=id_col)
dae_features = np.loadtxt(dae_feature_file, delimiter=',')
lgb_dae_predict_val = np.loadtxt(lgb_dae_predict_val_file)
lgb_dae_predict_tst = np.loadtxt(lgb_dae_predict_tst_file)
lgb_dae_te_predict_val = np.loadtxt(lgb_dae_te_predict_val_file)
lgb_dae_te_predict_tst = np.loadtxt(lgb_dae_te_predict_tst_file)
sdae_dae_predict_val = np.loadtxt(sdae_dae_predict_val_file)
sdae_dae_predict_tst = np.loadtxt(sdae_dae_predict_tst_file)

print(trn.shape, tst.shape, sub.shape, pseudo_label.shape, dae_features.shape)
print(lgb_dae_predict_val.shape, lgb_dae_predict_tst.shape)
print(lgb_dae_te_predict_val.shape, lgb_dae_te_predict_tst.shape)
print(sdae_dae_predict_val.shape, sdae_dae_predict_tst.shape)


# In[8]:


tst[target_col] = pseudo_label[target_col]
n_trn = trn.shape[0]
df = pd.concat([trn, tst], axis=0)
df.head()


# Loading 128 DAE features generated from [Supervised Emphasized Denoising AutoEncoder](https://www.kaggle.com/jeongyoonlee/supervised-emphasized-denoising-autoencoder/).

# In[9]:


df_dae = pd.DataFrame(dae_features, columns=[f'enc_{x}' for x in range(encoding_dim)])
print(df_dae.shape)
df_dae.head()


# Feature engineering using @udbhavpangotra's [code](https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model).

# In[10]:


# Feature engineering code from https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model

df['Embarked'] = df['Embarked'].fillna('No')
df['Cabin'] = df['Cabin'].fillna('_')
df['CabinType'] = df['Cabin'].apply(lambda x:x[0])
df.Ticket = df.Ticket.map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

df['Age'].fillna(round(df['Age'].median()), inplace=True,)
df['Age'] = df['Age'].apply(round).astype(int)

# Fare, fillna with mean value
fare_map = df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
df['Fare'] = df['Fare'].fillna(df['Pclass'].map(fare_map['Fare']))

df['FirstName'] = df['Name'].str.split(', ').str[0]
df['SecondName'] = df['Name'].str.split(', ').str[1]

df['n'] = 1

gb = df.groupby('FirstName')
df_names = gb['n'].sum()
df['SameFirstName'] = df['FirstName'].apply(lambda x:df_names[x]).fillna(1)

gb = df.groupby('SecondName')
df_names = gb['n'].sum()
df['SameSecondName'] = df['SecondName'].apply(lambda x:df_names[x]).fillna(1)

df['Sex'] = (df['Sex'] == 'male').astype(int)

df['FamilySize'] = df.SibSp + df.Parch + 1

feature_cols = ['Pclass', 'Age','Embarked','Parch','SibSp','Fare','CabinType','Ticket','SameFirstName', 'SameSecondName', 'Sex',
                'FamilySize', 'FirstName', 'SecondName']
cat_cols = ['Pclass','Embarked','CabinType','Ticket', 'FirstName', 'SecondName']
num_cols = [x for x in feature_cols if x not in cat_cols]
print(len(feature_cols), len(cat_cols), len(num_cols))


# Applying `log2(1 + x)` for numerical features and label-encoding categorical features using `kaggler.preprocessing.LabelEncoder`, which handles `NaN`s and groups rare categories together.

# In[11]:


for col in ['SameFirstName', 'SameSecondName', 'Fare', 'FamilySize', 'Parch', 'SibSp']:
    df[col] = np.log2(1 + df[col])
    
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

lbe = LabelEncoder(min_obs=50)
df[cat_cols] = lbe.fit_transform(df[cat_cols]).astype(int)


# # Part 2: Level-1 Base Model Training

# In[12]:


# Model params from https://www.kaggle.com/remekkinas/ensemble-learning-meta-classifier-for-stacking by remekkinas

lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': n_est,
    'objective': 'binary',
    'random_state': seed,
    'learning_rate': 0.01,
    'min_child_samples': 20,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 63,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}

ctb_params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': seed,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': n_est,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': seed
}


# In[13]:


base_models = {'rf': RandomForestClassifier(**rf_params), 
               'cbt': CatBoostClassifier(**ctb_params, verbose=None, logging_level='Silent'),
               'lgb': LGBMClassifier(**lgb_params),
               'et': ExtraTreesClassifier(bootstrap=True, criterion='entropy', max_features=0.55, min_samples_leaf=8, min_samples_split=4, n_estimators=100)}


# Make sure that you use the same CV folds across all level-1 models.

# In[14]:


from copy import copy

X = pd.concat((df[feature_cols], df_dae), axis=1)
y = df[target_col]
X_tst = X.iloc[n_trn:]

cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

p_dict = {}
for name in base_models:
    print(f'Training {name}:')
    p = np.zeros_like(y, dtype=float)
    p_tst = np.zeros((tst.shape[0],))
    for i, (i_trn, i_val) in enumerate(cv.split(X, y)):
        clf = copy(base_models[name])
        clf.fit(X.iloc[i_trn], y[i_trn])
        
        p[i_val] = clf.predict_proba(X.iloc[i_val])[:, 1]
        print(f'\tCV #{i + 1} AUC: {roc_auc_score(y[i_val], p[i_val]):.6f}')

    p_dict[name] = p
    print(f'\tCV AUC: {roc_auc_score(y, p):.6f}')


# Adding CV predictions of two additional models trained separately. You can use all models trained throughout the competition as long as those are traine d with the same CV folds.
# 
# **ALWAYS SAVE CV PREDICTIONS!!!**

# In[15]:


p_dict.update({
    'lgb_dae': lgb_dae_predict_val,
    'lgb_dae_te': lgb_dae_te_predict_val,
    'sdae_dae': sdae_dae_predict_val
})

dump(p_dict, 'predict_val_dict.joblib')


# # Part 3: Level-2 Stacking

# Training a level-2 LightGBM model with the level-1 model CV predictions, original features, and DAE features as inputs. If you have enough level-1 model predictions, you can train level-2 models only with level-1 model predictions. Here, since we only have six level-1 models, we use additional features and perform feature selection.

# In[16]:


X = pd.concat([pd.DataFrame(p_dict), df[feature_cols], df_dae], axis=1)
X_tst = X.iloc[n_trn:]

p = np.zeros_like(y, dtype=float)
p_tst = np.zeros((tst.shape[0],))
print(f'Training a stacking ensemble LightGBM model:')
for i, (i_trn, i_val) in enumerate(cv.split(X, y)):
    if i == 0:
        clf = AutoLGB(objective='binary', metric='auc', sample_size=len(i_trn), random_state=seed)
        clf.tune(X.iloc[i_trn], y[i_trn])
        features = clf.features
        params = clf.params
        n_best = clf.n_best
        print(f'{n_best}')
        print(f'{params}')
        print(f'{features}')
    
    trn_data = lgb.Dataset(X.iloc[i_trn], y[i_trn])
    val_data = lgb.Dataset(X.iloc[i_val], y[i_val])
    clf = lgb.train(params, trn_data, n_best, val_data, verbose_eval=100)
    p[i_val] = clf.predict(X.iloc[i_val])
    p_tst += clf.predict(X_tst) / n_fold
    print(f'CV #{i + 1} AUC: {roc_auc_score(y[i_val], p[i_val]):.6f}')


# In[17]:


print(f'  CV AUC: {roc_auc_score(y, p):.6f}')
print(f'Test AUC: {roc_auc_score(pseudo_label[target_col], p_tst)}')


# In[18]:


n_pos = int(0.34911 * tst.shape[0])
th = sorted(p_tst, reverse=True)[n_pos]
print(th)
confusion_matrix(pseudo_label[target_col], (p_tst > th).astype(int))


# In[19]:


sub[target_col] = (p_tst > th).astype(int)
sub.to_csv(submission_file)


# If you find it useful, please upvote the notebook and leave your feedback. It will be greatly appreciated!
# 
# Also please check out my previous notebooks as follows:
# * [AutoEncoder + Pseudo Label + AutoLGB](https://www.kaggle.com/jeongyoonlee/autoencoder-pseudo-label-autolgb): shows how to build a basic AutoEncoder using Keras, and perform automated feature selection and hyperparameter optimization using `Kaggler`'s `AutoLGB`.
# * [Supervised Emphasized Denoising AutoEncoder](https://www.kaggle.com/jeongyoonlee/supervised-emphasized-denoising-autoencoder): shows how to build a more sophiscated version of AutoEncoder, called supervised emphasized Denoising AutoEncoder (DAE), which trains DAE and a classifier simultaneously.
# 
# Happy Kaggling! ;)
# 

# In[ ]:




