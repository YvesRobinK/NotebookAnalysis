#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will show how to use autoencoder, feature selection, hyperparameter optimization, and pseudo labeling using the `Keras` and `Kaggler` Python packages.
# 
# The contents of the notebook are as follows:
# 1. **Package installation**: Installing latest version of `Kaggler` using `Pip`
# 2. **Regular feature engineering**: [code](https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model) by @udbhavpangotra
# 3. **Feature transformation**: Using `kaggler.preprocessing.LabelEncoder` to impute missing values and group rare categories automatically.
# 4. **Stacked AutoEncoder**: Notebooks for DAE will be shared later.
# 5. **Model training**: with 5-fold CV and pseudo label from @hiro5299834's [data](https://www.kaggle.com/hiro5299834/tps-apr-2021-voting-pseudo-labeling).
# 6. **Feature selection and hyperparameter optimization**: Using `kaggler.model.AutoLGB`
# 7. **Saving a submission file**

# ## Load libraries and install `Kaggler`

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import lightgbm as lgb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings


# In[3]:


get_ipython().system('pip install kaggler')


# In[4]:


import kaggler
from kaggler.model import AutoLGB
from kaggler.preprocessing import LabelEncoder

print(f'Kaggler: {kaggler.__version__}')
print(f'TensorFlow: {tf.__version__}')


# In[5]:


warnings.simplefilter('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)


# ## Feature Engineering (ref: [code](https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model) by @udbhavpangotra)

# In[6]:


feature_name = 'ae'
algo_name = 'lgb'
model_name = f'{algo_name}_{feature_name}'

data_dir = Path('/kaggle/input/tabular-playground-series-apr-2021/')
trn_file = data_dir / 'train.csv'
tst_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'
pseudo_label_file = '/kaggle/input/tps-apr-2021-label/voting_submission_from_5_best.csv'

feature_file = f'{feature_name}.csv'
predict_val_file = f'{model_name}.val.txt'
predict_tst_file = f'{model_name}.tst.txt'
submission_file = f'{model_name}.sub.csv'

target_col = 'Survived'
id_col = 'PassengerId'


# In[7]:


trn = pd.read_csv(trn_file, index_col=id_col)
tst = pd.read_csv(tst_file, index_col=id_col)
sub = pd.read_csv(sample_file, index_col=id_col)
pseudo_label = pd.read_csv(pseudo_label_file, index_col=id_col)
print(trn.shape, tst.shape, sub.shape, pseudo_label.shape)


# In[8]:


tst[target_col] = pseudo_label[target_col]
n_trn = trn.shape[0]
df = pd.concat([trn, tst], axis=0)
df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.nunique()


# In[12]:


# Feature engineering code from https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model

df['Embarked'] = df['Embarked'].fillna('No')
df['Cabin'] = df['Cabin'].fillna('_')
df['CabinType'] = df['Cabin'].apply(lambda x:x[0])
df.Ticket = df.Ticket.map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

df['Age'].fillna(round(df['Age'].median()), inplace=True,)
df['Age'] = df['Age'].apply(round).astype(int)

df['Fare'].fillna(round(df['Fare'].median()), inplace=True,)

df['FirstName'] = df['Name'].str.split(', ').str[0]
df['SecondName'] = df['Name'].str.split(', ').str[1]

df['n'] = 1

gb = df.groupby('FirstName')
df_names = gb['n'].sum()
df['SameFirstName'] = df['FirstName'].apply(lambda x:df_names[x])

gb = df.groupby('SecondName')
df_names = gb['n'].sum()
df['SameSecondName'] = df['SecondName'].apply(lambda x:df_names[x])

df['Sex'] = (df['Sex'] == 'male').astype(int)

df['FamilySize'] = df.SibSp + df.Parch + 1

feature_cols = ['Pclass', 'Age','Embarked','Parch','SibSp','Fare','CabinType','Ticket','SameFirstName', 'SameSecondName', 'Sex',
                'FamilySize', 'FirstName', 'SecondName']
cat_cols = ['Pclass','Embarked','CabinType','Ticket', 'FirstName', 'SecondName']
num_cols = [x for x in feature_cols if x not in cat_cols]
print(len(feature_cols), len(cat_cols), len(num_cols))


# In[13]:


df[num_cols].describe()


# In[14]:


plt.figure(figsize=(16, 16))
for i, col in enumerate(num_cols):
    ax = plt.subplot(4, 2, i + 1)
    ax.set_title(col)
    df[col].hist(bins=50)


# ## Feature Transformation

# Apply `log2(1 + x)` transformation followed by standardization for count variables to make them close to the normal distribution. `log2(1 + x)` has better resolution than `log1p` and it preserves the values of 0 and 1.

# In[15]:


for col in ['SameFirstName', 'SameSecondName', 'Fare', 'FamilySize', 'Parch', 'SibSp']:
    df[col] = np.log2(1 + df[col])
    
df.describe()


# In[16]:


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# Label-encode categorical variables using `kaggler.preprocessing.LabelEncoder`, which creates new categories for `NaN`s as well as rare categories (using the threshold of `min_obs`).

# In[17]:


lbe = LabelEncoder(min_obs=50)
df[cat_cols] = lbe.fit_transform(df[cat_cols]).astype(int)


# ## AutoEncoder using `Keras`

# Basic stacked autoencoder. I will add the versions with DAE and emphasized DAE later.

# In[18]:


encoding_dim = 64

def get_model(encoding_dim, dropout=.2):
    num_dim = len(num_cols)
    num_input = keras.layers.Input((num_dim,), name='num_input')
    cat_inputs = []
    cat_embs = []
    emb_dims = 0
    for col in cat_cols:
        cat_input = keras.layers.Input((1,), name=f'{col}_input')
        emb_dim = max(8, int(np.log2(1 + df[col].nunique()) * 4))
        cat_emb = keras.layers.Embedding(input_dim=df[col].max() + 1, output_dim=emb_dim)(cat_input)
        cat_emb = keras.layers.Dropout(dropout)(cat_emb)
        cat_emb = keras.layers.Reshape((emb_dim,))(cat_emb)

        cat_inputs.append(cat_input)
        cat_embs.append(cat_emb)
        emb_dims += emb_dim

    merged_inputs = keras.layers.Concatenate()([num_input] + cat_embs)

    encoded = keras.layers.Dense(encoding_dim * 3, activation='relu')(merged_inputs)
    encoded = keras.layers.Dropout(dropout)(encoded)
    encoded = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    encoded = keras.layers.Dropout(dropout)(encoded)    
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
    
    decoded = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = keras.layers.Dropout(dropout)(decoded)
    decoded = keras.layers.Dense(encoding_dim * 3, activation='relu')(decoded)
    decoded = keras.layers.Dropout(dropout)(decoded)    
    decoded = keras.layers.Dense(num_dim + emb_dims, activation='linear')(decoded)

    encoder = keras.Model([num_input] + cat_inputs, encoded)
    ae = keras.Model([num_input] + cat_inputs, decoded)
    ae.add_loss(keras.losses.mean_squared_error(merged_inputs, decoded))
    ae.compile(optimizer='adam')
    return ae, encoder


# In[19]:


ae, encoder = get_model(encoding_dim)
ae.summary()


# In[20]:


inputs = [df[num_cols].values] + [df[x].values for x in cat_cols]
ae.fit(inputs, inputs,
      epochs=100,
      batch_size=16384,
      shuffle=True,
      validation_split=.2)


# In[21]:


encoding = encoder.predict(inputs)
print(encoding.shape)
np.savetxt(feature_file, encoding, fmt='%.6f', delimiter=',')


# ## Model Training + Feature Selection + Hyperparameter Optimization

# Train the `LightGBM` model with pseudo label and 5-fold CV. In the first fold, perform feature selection and hyperparameter optimization using `kaggler.model.AutoLGB`.

# In[22]:


seed = 42
n_fold = 5
X = pd.concat((df[feature_cols], 
               pd.DataFrame(encoding, columns=[f'enc_{x}' for x in range(encoding_dim)])), axis=1)
y = df[target_col]
X_tst = X.iloc[n_trn:]

cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
p = np.zeros_like(y, dtype=float)
p_tst = np.zeros((tst.shape[0],))
for i, (i_trn, i_val) in enumerate(cv.split(X, y)):
    if i == 0:
        clf = AutoLGB(objective='binary', metric='auc', random_state=seed)
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

np.savetxt(predict_val_file, p, fmt='%.6f')
np.savetxt(predict_tst_file, p_tst, fmt='%.6f')


# In[23]:


print(f'  CV AUC: {roc_auc_score(y, p):.6f}')
print(f'Test AUC: {roc_auc_score(pseudo_label[target_col], p_tst)}')


# ## Submission File

# In[24]:


n_pos = int(0.34911 * tst.shape[0])
th = sorted(p_tst, reverse=True)[n_pos]
print(th)
confusion_matrix(pseudo_label[target_col], (p_tst > th).astype(int))


# In[25]:


sub[target_col] = (p_tst > th).astype(int)
sub.to_csv(submission_file)


# If you find it helpful, please upvote the notebook and give a star to [Kaggler](https://github.com/jeongyoonlee/Kaggler). If you have questions and/or feature requests for Kaggler, please post them as `Issue` in the `Kaggler` GitHub repository.
# 
# Happy Kaggling!

# In[ ]:




