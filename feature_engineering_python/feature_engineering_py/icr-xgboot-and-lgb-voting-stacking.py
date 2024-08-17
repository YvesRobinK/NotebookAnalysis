#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as ct
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Set up Kaggle dataset directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


def balanced_log_loss(y_true, y_pred):
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0))
    log_loss_1 = -np.sum(y_true * np.log(p_1))
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    balanced_log_loss = 2*(w_0 * log_loss_0 + w_1 * log_loss_1) / (w_0 + w_1)
    return balanced_log_loss/(N_0+N_1)


# ## **Importing data**

# In[3]:


train = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/train.csv')
test = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
greeks = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/greeks.csv')
sample_submission = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')


# In[4]:


train.head()


# In[5]:


greeks.head()


# In[6]:


train.loc[:, train.isna().mean()>0] .isna().mean()


# In[7]:


df = pd.merge(train, greeks, how='left', on='Id')


# In[8]:


df.head()


# In[9]:


df.describe().T


# ## **Data Cleaning**

# In[10]:


df['Class'].value_counts()


# In[11]:


df['Alpha'].value_counts()


# In[12]:


def encode(dataframe):
    le = LabelEncoder()
    obj = list(dataframe.loc[:, dataframe.dtypes == 'object'].columns)
    for i in obj:
        if i not in ['Id', 'Epsilon']:
            dataframe[i] = le.fit_transform(dataframe[i])
    return dataframe


# In[13]:


df = encode(df)  
test = encode(test)


# In[14]:


df.columns


# In[15]:


features = ['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN',
       'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS',
       'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY',
       'EB', 'EE', 'EG', 'EH', 'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI',
       'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL']

target = 'Class'


# In[16]:


imputer = KNNImputer(n_neighbors=2)

df[features] = imputer.fit_transform(df[features])
test[features] = imputer.fit_transform(test[features])


# In[17]:


df.head()


# ## **Vizualization**

# In[18]:


correlation_matrix = df.corr().abs()

correlation_matrix = correlation_matrix[correlation_matrix < 1.0]

top_correlated_features = correlation_matrix.unstack().sort_values(ascending=False)[:100]

print(top_correlated_features)


# In[19]:


fig = plt.figure(figsize=(6*6, 45), dpi=130)
for idx, col in enumerate(features):
    ax = plt.subplot(19, 3, idx + 1)
    sns.kdeplot(
        data=df, hue='Class', fill=True,
        x=col,legend=False
    )
            
    ax.set_ylabel(''); ax.spines['top'].set_visible(False), 
    ax.set_xlabel(''); ax.spines['right'].set_visible(False)
    ax.set_title(f'{col}', loc='right', 
                 weight='bold', fontsize=20)

fig.suptitle(f'Features vs Class\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.97), fontsize=25, ncol=3)
plt.tight_layout()
plt.show()


# ## **Feature Engineering**

# In[20]:


#soon


# ## **Modeling**

# In[21]:


df.columns


# In[22]:


features = [ 'AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD ', 'BN',
       'BP', 'BQ', 'BR', 'BZ', 'CB', 'CC', 'CD ', 'CF', 'CH', 'CL', 'CR', 'CS',
       'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DI', 'DL', 'DN', 'DU', 'DV', 'DY',
       'EB', 'EE', 'EG', 'EH', 'EJ', 'EL', 'EP', 'EU', 'FC', 'FD ', 'FE', 'FI',
       'FL', 'FR', 'FS', 'GB', 'GE', 'GF', 'GH', 'GI', 'GL'
        ]

target = 'Class'


# In[23]:


get_ipython().run_cell_magic('capture', '', "\nxgb_params = {\n    'colsample_bytree': 0.5, \n    'gamma': 1.0,\n    'learning_rate': 0.01777187034634523,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    'n_estimators': 1500, \n    'subsample': 0.7629766636827013,\n    'verbosity': 0,\n    'random_state': 42,\n    'tree_method': 'gpu_hist',\n    'predictor': 'gpu_predictor'\n}\n\nxgb_params1 = {\n    'colsample_bytree': 0.5,\n    'gamma': 0.08898545568136436,\n    'learning_rate': 0.009253274006068297,\n    'max_depth': 3,\n    'min_child_weight': 1,\n    'n_estimators': 1500,\n    'subsample': 0.8971494956585011,\n    'verbosity': 0,\n    'random_state': 42,\n    'tree_method': 'gpu_hist',\n    'predictor': 'gpu_predictor'\n}\n\nlgb_params = {\n    'colsample_bytree': 0.5, \n    'learning_rate': 0.02, \n    'max_depth': 8, \n    'min_child_samples': 57,\n    'n_estimators': 2445, \n    'num_leaves': 57,\n    'reg_alpha': 0.6197994214239195, \n    'reg_lambda': 0.8675671389814725, \n    'subsample': 0.5264255077986388,\n    'device': 'gpu',\n    'random_state': 42,\n    'verbose': -1 \n}\n\nlgb1_params = {\n    'colsample_bytree': 0.5, \n    'learning_rate': 0.02,\n    'max_depth': 4,\n    'min_child_samples': 5,\n    'n_estimators': 3000,\n    'num_leaves': 100,\n    'reg_alpha': 1.0,\n    'reg_lambda': 1.0,\n    'subsample': 0.5,\n    'device': 'gpu',\n    'random_state': 42,\n    'verbose': -1 \n} \n\nlgb2_params = {\n    'colsample_bytree': 0.5,\n    'learning_rate': 0.02,\n    'max_depth': 4,\n    'min_child_samples': 5,\n    'n_estimators': 1476,\n    'num_leaves': 100,\n    'reg_alpha':  0.6362952390423132,\n    'reg_lambda': 1.0, \n    'subsample': 1.0,\n    'device': 'gpu',\n    'random_state': 42,\n    'verbose': -1 \n} \n\n\nmodels = [\n    ('xgb', xgb.XGBClassifier(**xgb_params)),\n    ('xgb1', xgb.XGBClassifier(**xgb_params1)),\n    ('lgb', lgb.LGBMClassifier(**lgb_params)),\n    ('lgb1', lgb.LGBMClassifier(**lgb1_params)),\n    ('lgb2', lgb.LGBMClassifier(**lgb2_params))\n]\n\nstacking_model = StackingClassifier(\n        estimators=models[1:],\n        final_estimator=xgb.XGBClassifier(**xgb_params),\n        cv=5,\n        stack_method='predict_proba',\n        n_jobs=-1\n)\n\nvoting_model = VotingClassifier(models, voting='soft')\n\nmodels_iter = {\n    'stacking': stacking_model,\n    'voting': voting_model\n}\n\n# Load your dataset\nX = df[features]\ny = df[target]\n\nskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\nscores = []\nm = []\n\nfor train_idx, val_idx in skf.split(X, y):\n    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]\n    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]\n\n    for model in models_iter.values():\n        \n        pipeline = Pipeline([\n            ('scaler', MinMaxScaler()),\n            ('model', model)\n        ])\n        pipeline.fit(X_train, y_train)\n        val_preds = pipeline.predict_proba(X_valid)\n        val_score = balanced_log_loss(y_valid, val_preds[:, 1])\n        m.append(pipeline)\n        scores.append(val_score)\n")


# In[24]:


print('*' * 45)
print(f'Log-loss scores: {scores}')
print('*' * 45)
print(f'Log-loss scores mean: {np.mean(scores)}')


# ## **Submission**

# In[25]:


sample_submission.head()


# In[26]:


prediction = [0,0]
for model in m:
    prediction += model.predict_proba(test[features])


# In[27]:


# sample_submission[['class_0', 'class_1', 'class_2', 'class_3']] = prediction/len(m)
# sample_submission['class_1'] = sample_submission['class_1'] + sample_submission['class_2'] + sample_submission['class_3']
# sample_submission = sample_submission.drop(['class_2', 'class_3'], axis=1)


# In[28]:


sample_submission[['class_0', 'class_1']] = prediction/len(m)


# In[29]:


sample_submission.head()


# In[30]:


sample_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




