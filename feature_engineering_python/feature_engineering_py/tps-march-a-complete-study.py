#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from colorama import Fore

from pandas_profiling import ProfileReport
import seaborn as sns
from sklearn import metrics
from scipy import stats
import math

from tqdm.notebook import tqdm
from copy import deepcopy

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import optuna
from optuna import Trial, visualization

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score, mean_squared_error


# In[2]:


# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"

primary_green = px.colors.qualitative.Plotly[2]


# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

meta_random_seed = 68

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


# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:180%; text-align:center">Tabular Playground Series 📚 - March 2021 📈</p>
# 
# ![kaggle-python.png](attachment:kaggle-python.png)

# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">Table of Content</p>
# 
# * [1. Data visualization 📊](#1)
#     * [1.1 Target](#1.1)
#     * [1.2 Numerical Columns](#1.2)
#     * [1.3 Categorical Columns](#1.3)
# * [2. Feature Engineering 🔧](#2)
# * [3. Base Model ⚙️](#3)
#     * [3.1 XGBoost](#3.1)
#     * [3.2 LGBM](#3.2)
# * [4. Essemble models 🏂](#4)
#     * [4.1 L1 classification 📝](#4.1)
#     * [4.2 Blended classification 📈](#4.2)
#     * [4.3 L2 classification 📝](#4.3)
# * [5. Models Optimization with Optuna ⛷](#5)
#     * [5.1 L1 classification 📝](#5.1)
#     * [5.2 L1 classification 📝](#5.2)
# * [6. H2O AutoML 🧮](#6)
#     * [6.1 H2O AutoML Submission 📝](#6.1)
#     * [6.2 L2 AutoML Classification](#6.2)
#     * [6.3 L2 Optimized + AutoML leader Classification](#6.3)
# * [7. Fianl Submission](#7)

# In[4]:


train_df = pd.read_csv('/kaggle/input/tabular-playground-series-mar-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-mar-2021/test.csv')
sub_df = pd.read_csv('/kaggle/input/tabular-playground-series-mar-2021/sample_submission.csv')

train_df.head()


# In[5]:


feature_cols = train_df.drop(['id', 'target'], axis=1).columns

## Getting all the data that are not of "object" type. 
numerical_columns = train_df[feature_cols].select_dtypes(include=['int64','float64']).columns
categorical_columns = train_df[feature_cols].select_dtypes(exclude=['int64','float64']).columns

print(len(numerical_columns), len(categorical_columns))


# In[6]:


## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_indexs = train_df.index
test_indexs = test_df.index

df =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)
df = df.drop(['id', 'target'], axis=1)

print(df.shape)


# <a id='1'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center">1. Data visualization 📊</p>

# In[7]:


profile = ProfileReport(train_df)


# In[8]:


profile


# <a id='1.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">1.1 Target Variable</p>

# In[9]:


fig = px.histogram(train_df, x='target')
fig.update_layout(
    title_text='Target distribution', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
)
fig.show()


# <a id='1.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">1.2 Numerical Variables</p>

# In[10]:


num_rows, num_cols = 4,3
f, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))
f.suptitle('Distribution of Features', fontsize=16)

for index, column in enumerate(df[numerical_columns].columns):
    i,j = (index // num_cols, index % num_cols)
    sns.kdeplot(train_df.loc[train_df['target'] == 0, column], color="m", shade=True, ax=axes[i,j])
    sns.kdeplot(train_df.loc[train_df['target'] == 1, column], color="b", shade=True, ax=axes[i,j])

f.delaxes(axes[3, 2])
plt.tight_layout()
plt.show()


# In[11]:


corr = df[numerical_columns].corr().abs()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

fig, ax = plt.subplots(figsize=(14, 14))

# plot heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            cbar_kws={"shrink": .8}, vmin=0, vmax=1)
# yticks
plt.yticks(rotation=0)
plt.show()


# As we can see, the numerical columns are low correlated between them and the distribution of all of them doesn't variate over target value.

# In[12]:


# Thanks a lot @dwin183287 for sharing this amazinf function!

background_color = "#f6f5f5"

fig = plt.figure(figsize=(12, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])

ax0.set_facecolor(background_color)
ax0.text(-1.1, 0.26, 'Correlation of Continuous Features with Target', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-1.1, 0.24, 'There is no features that pass 0.22 correlation with target', fontsize=13, fontweight='light', fontfamily='serif')

chart_df = pd.DataFrame(train_df[numerical_columns].corrwith(train_df['target']))
chart_df.columns = ['corr']
sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, color=primary_blue, zorder=3, edgecolor='black', linewidth=1.5)
ax0.grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_ylabel('')

for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

plt.show()


# <a id='1.3'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">1.3 Categorical Variables</p>

# In[13]:


train_0_df = train_df.loc[train_df['target'] == 0]
train_1_df = train_df.loc[train_df['target'] == 1]

num_rows, num_cols = 5,4
fig = make_subplots(rows=num_rows, cols=num_cols)

for index, column in enumerate(df[categorical_columns].columns):
    i,j = ((index // num_cols)+1, (index % num_cols)+1)
    data = train_0_df.groupby(column)[column].count().sort_values(ascending=False)
    data = data if len(data) < 10 else data[:10]
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 0',
    ), row=i, col=j)

    data = train_1_df.groupby(column)[column].count().sort_values(ascending=False)
    data = data if len(data) < 10 else data[:10]
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 1'
    ), row=i, col=j)
    
    fig.update_xaxes(title=column, row=i, col=j)
    fig.update_layout(barmode='stack')
    
fig.update_layout(
    autosize=False,
    width=1600,
    height=1600,
    showlegend=False,
)
fig.show()


# As we can see, many categorical columns has values that only apply to one category (0 or 1) so we can engineer this variables.

# ### High cardinality variables

# In[14]:


num_rows, num_cols = 10,1
fig = make_subplots(rows=num_rows, cols=num_cols)
cont = 1

for index, column in enumerate(df[categorical_columns].columns):
    data = train_0_df.groupby(column)[column].count().sort_values(ascending=False)
    if len(data) < 10:
        continue
    # data = data if len(data) < 25 else data[:25]
    i,j = (cont, 1)
    cont+=1
    
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 0',
    ), row=i, col=j)
    
    target_0_values = set(deepcopy(data.index))
    
    data = train_1_df.groupby(column)[column].count().sort_values(ascending=False)
    # data = data if len(data) < 25 else data[:25]
    
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 1'
    ), row=i, col=j)
    
    target_1_values = set(deepcopy(data.index))
    
    print('----------------------{}----------------------'.format(column))
    print('Unique values for class 0: {}'.format(target_0_values - target_1_values))
    print('Unique values for class 1: {}'.format(target_1_values - target_0_values))
    
    fig.update_xaxes(title=column, row=i, col=j)
    fig.update_layout(barmode='stack')
    
fig.update_layout(
    autosize=False,
    width=900,
    height=2000,
    showlegend=False,
)
fig.show()


# Lets analyze the results:
# - **cat1**: We can merge D and E as they have really low representation
# - **cat2**: We can merge all values diferent than: ['A', 'C', 'D', 'G', 'F', 'J', 'I', 'M', 'Q', 'L', 'O'] as they have really low representation.
# - **cat3**: ['I', 'L', 'K'] and ['H', 'J', 'G'] can be merged.
# - **cat4**: ['C', 'S', 'T', 'R'] can be merged into 'C' and others than ['E', 'F', 'D', 'G', 'H', 'J', 'K', 'I', 'C'] can be deleted
# - **cat5**: Other than ['BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N'] can be deleted.
# - **cat7**: ['Y', 'AA', 'R', 'O', 'AP', 'AY'] and ['AL', 'V', 'BA', 'AC', 'AD', 'L'] can be merged.
# - **cat10**: ['GE', 'LN', 'HJ', 'IG', 'EK', 'HB', 'DF'] and ['CD', 'GI', 'HC', 'JR', 'MC', 'FR', 'GK'] can be merged.

# <a id='2'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center">2. Feature Engineering 🔧</p>

# ### Aggregate values

# In[15]:


# Fix cat5 variable
train_df['cat5'] = train_df['cat5'].apply(lambda x: x if x not in ['AK', 'BX', 'BP', 'AG', 'BM', 'CB', 'B', 'ZZ'] else 'CAT50')
test_df['cat5'] = test_df['cat5'].apply(lambda x: x if x not in ['AK', 'BX', 'BP', 'AG', 'BM', 'CB', 'B', 'ZZ'] else 'CAT50')

# Fix cat8 variable
train_df['cat8'] = train_df['cat8'].apply(lambda x: x if x not in ['P', 'AC'] else 'CAT80')
test_df['cat8'] = test_df['cat8'].apply(lambda x: x if x not in ['P', 'AC'] else 'CAT80')

# Fix cat10 variable
train_df['cat10'] = train_df['cat10'].apply(lambda x: x if x not in ['AF', 'MR', 'DU', 'AW', 'DL', 'GJ', 'MK', 'MA', 'DT', 'FA', 'GY', 'EN', 'EH', 'JE', 'JF', 'KK', 'LH', 'LK', 'MW', 'FF', 'CH', 'JU', 'HY', 'LR', 'KI', 'IU', 'CM', 'DM', 'BD', 'MU', 'ML', 'EB', 'IQ', 'CF', 'IN', 'CN', 'IM', 'AJ', 'IP', 'MI', 'ED', 'CX', 'FW', 'BS', 'IY', 'MP', 'BX', 'DN', 'MO', 'GH', 'EG', 'BA', 'ME', 'GR', 'KD', 'LT', 'IL', 'GF', 'BO', 'DA', 'MQ', 'KU', 'DX', 'CT', 'HF', 'CQ', 'GG', 'EF', 'HI', 'KN', 'GV', 'JC', 'DK', 'GD'] else 'CAT10')
test_df['cat10'] = test_df['cat10'].apply(lambda x: x if x not in ['AF', 'MR', 'DU', 'AW', 'DL', 'GJ', 'MK', 'MA', 'DT', 'FA', 'GY', 'EN', 'EH', 'JE', 'JF', 'KK', 'LH', 'LK', 'MW', 'FF', 'CH', 'JU', 'HY', 'LR', 'KI', 'IU', 'CM', 'DM', 'BD', 'MU', 'ML', 'EB', 'IQ', 'CF', 'IN', 'CN', 'IM', 'AJ', 'IP', 'MI', 'ED', 'CX', 'FW', 'BS', 'IY', 'MP', 'BX', 'DN', 'MO', 'GH', 'EG', 'BA', 'ME', 'GR', 'KD', 'LT', 'IL', 'GF', 'BO', 'DA', 'MQ', 'KU', 'DX', 'CT', 'HF', 'CQ', 'GG', 'EF', 'HI', 'KN', 'GV', 'JC', 'DK', 'GD'] else 'CAT10')


# In[16]:


# try to delete low represented vars

train_df['cat4'] = train_df['cat4'].apply(lambda x: x if x in ['E', 'F', 'D', 'G', 'H', 'J', 'K', 'I', 'C'] else 'Z')
test_df['cat4'] = test_df['cat4'].apply(lambda x: x if x in ['E', 'F', 'D', 'G', 'H', 'J', 'K', 'I', 'C'] else 'Z')

train_df['cat5'] = train_df['cat5'].apply(lambda x: x if x in ['BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N', 'CL', 'CAT50'] else 'Z')
test_df['cat5'] = test_df['cat5'].apply(lambda x: x if x in ['BI', 'AB', 'BU', 'K', 'G', 'BQ', 'N', 'CL', 'CAT50'] else 'Z')


# In[17]:


train_0_df = train_df.loc[train_df['target'] == 0]
train_1_df = train_df.loc[train_df['target'] == 1]

num_rows, num_cols = 10,1
fig = make_subplots(rows=num_rows, cols=num_cols)
cont = 1

for index, column in enumerate(df[categorical_columns].columns):
    data = train_0_df.groupby(column)[column].count().sort_values(ascending=False)
    if len(data) < 10:
        continue
    data = data if len(data) < 25 else data[:25]
    i,j = (cont, 1)
    cont+=1
    
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 0',
    ), row=i, col=j)
        
    data = train_1_df.groupby(column)[column].count().sort_values(ascending=False)
    data = data if len(data) < 25 else data[:25]
    
    fig.add_trace(go.Bar(
        x = data.index,
        y = data.values,
        name='Label: 1'
    ), row=i, col=j)
        
    fig.update_xaxes(title=column, row=i, col=j)
    fig.update_layout(barmode='stack')
    
fig.update_layout(
    autosize=False,
    width=900,
    height=2000,
    showlegend=False,
)
fig.show()


# ### Label Encoder
# 
# ![LabelEncoder.png](attachment:LabelEncoder.png)

# In[18]:


from category_encoders import CatBoostEncoder, LeaveOneOutEncoder
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

loo_features = []
le_features = []

    
def label_encoder(train_df, test_df, column):
    le = LabelEncoder()
    new_feature = "{}_le".format(column)
    le.fit(train_df[column].unique().tolist() + test_df[column].unique().tolist())
    
    train_df[new_feature] = le.transform(train_df[column])
    test_df[new_feature] = le.transform(test_df[column])
    
    return new_feature

def loo_encode(train_df, test_df, column):
    loo = LeaveOneOutEncoder()
    new_feature = "{}_loo".format(column)
    loo.fit(train_df[column], train_df["target"])
    
    train_df[new_feature] = loo.transform(train_df[column])
    test_df[new_feature] = loo.transform(test_df[column])
    
    return new_feature

for feature in categorical_columns:
    loo_features.append(loo_encode(train_df, test_df, feature))
    le_features.append(label_encoder(train_df, test_df, feature))
    
xgb_cat_features = deepcopy(loo_features)
lgb_cat_features = deepcopy(le_features)
cb_cat_features = deepcopy(list(categorical_columns))
ridge_cat_features = deepcopy(loo_features)


# ### Standarize numerical features

# In[ ]:





# <a id='3'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center">3. Base Model ⚙️</p>

# In[19]:


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score


# In[20]:


y = train_df["target"]
lgbm_features = lgb_cat_features + list(numerical_columns)
x = train_df[lgbm_features]

x_train, x_valid, y_train, y_valid=train_test_split(x, y, test_size=0.2, random_state=meta_random_seed)


# <a id='3.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">3.1 XGBoost</p>

# In[21]:


xgb_cls = XGBClassifier()

xgb_cls.fit(x_train, y_train, verbose=False)
predictions = xgb_cls.predict_proba(x_valid)[:,1]

auc = roc_auc_score(y_valid, predictions)

print(f'Baseline Score: {auc}')


# <a id='3.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">3.2 LGBM</p>

# In[22]:


lgbm = LGBMClassifier()

lgbm.fit(x_train, y_train, eval_set=(x_valid,y_valid), early_stopping_rounds=150, verbose=False)
predictions = lgbm.predict_proba(x_valid)[:,1]

auc = roc_auc_score(y_valid, predictions)

print(f'Baseline Score: {auc}')


# <a id='4'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center">4. Essemble models 🏂</p>

# In[ ]:


# I took many ideas from @craigmthomas, so thank you very much bro, your ideas are awesome

import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import StratifiedKFold


# In[ ]:


random_state = meta_random_seed
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

y = train_df["target"]

xgb_train_preds = np.zeros(len(train_df.index), )
xgb_test_preds = np.zeros(len(test_df.index), )
xgb_features = xgb_cat_features + list(numerical_columns)

lgbm_train_preds = np.zeros(len(train_df.index), )
lgbm_test_preds = np.zeros(len(test_df.index), )
lgbm_features = lgb_cat_features + list(numerical_columns)

cb_train_preds = np.zeros(len(train_df.index), )
cb_test_preds = np.zeros(len(test_df.index), )
cb_features = cb_cat_features + list(numerical_columns)

ridge_train_preds = np.zeros(len(train_df.index), )
ridge_test_preds = np.zeros(len(test_df.index), )
ridge_features = ridge_cat_features + list(numerical_columns)


# <a id='4.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">4.1 L1 classification</p>
# 
# This will be the first level classification, where many algorithms will predict in the train and test sets. Wi will then essemble them and try a second level classification.

# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\nfor fold, (train_index, test_index) in enumerate(k_fold.split(train_df, y)):\n    print("--> Fold {}".format(fold + 1))\n    y_train = y.iloc[train_index]\n    y_valid = y.iloc[test_index]\n\n    ########## Generate train and valid sets ##########\n    xgb_x_train = pd.DataFrame(train_df[xgb_features].iloc[train_index])\n    xgb_x_valid = pd.DataFrame(train_df[xgb_features].iloc[test_index])\n\n    lgbm_x_train = pd.DataFrame(train_df[lgbm_features].iloc[train_index])\n    lgbm_x_valid = pd.DataFrame(train_df[lgbm_features].iloc[test_index])\n    \n    cb_x_train = pd.DataFrame(train_df[cb_features].iloc[train_index])\n    cb_x_valid = pd.DataFrame(train_df[cb_features].iloc[test_index])\n\n    ridge_x_train = pd.DataFrame(train_df[ridge_features].iloc[train_index])\n    ridge_x_valid = pd.DataFrame(train_df[ridge_features].iloc[test_index])\n\n    ########## XGBoost model ##########\n    xgb_model = XGBClassifier(\n        seed=random_state,\n        verbosity=1,\n        eval_metric="auc",\n        tree_method="gpu_hist",\n        gpu_id=0,\n        n_jobs = 12,\n    )\n    xgb_model.fit(\n        xgb_x_train,\n        y_train,\n        eval_set=[(xgb_x_valid, y_valid)], \n        verbose=0,\n        early_stopping_rounds=200\n    )\n\n    train_oof_preds = xgb_model.predict_proba(xgb_x_valid)[:,1]\n    test_oof_preds = xgb_model.predict_proba(test_df[xgb_features])[:,1]\n    xgb_train_preds[test_index] = train_oof_preds\n    xgb_test_preds += test_oof_preds / n_folds\n    \n    print(": XGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))\n    \n    ########## LGBM model ##########\n    lgbm_model = LGBMClassifier(\n        cat_feature=[x for x in range(len(categorical_columns))],\n        random_state=random_state,\n        metric="auc",\n        n_jobs=12,\n    )\n    lgbm_model.fit(\n        lgbm_x_train,\n        y_train,\n        eval_set=[(lgbm_x_valid, y_valid)], \n        verbose=0,\n    )\n\n    train_oof_preds = lgbm_model.predict_proba(lgbm_x_valid)[:,1]\n    test_oof_preds = lgbm_model.predict_proba(test_df[lgbm_features])[:,1]\n    lgbm_train_preds[test_index] = train_oof_preds\n    lgbm_test_preds += test_oof_preds / n_folds\n    \n    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))\n\n    ########## CatBoost model ##########\n    cb_model = CatBoostClassifier(\n        verbose=0,\n        eval_metric="AUC",\n        loss_function="Logloss",\n        random_state=random_state,\n        task_type="GPU",\n        devices="0",\n        cat_features=[x for x in range(len(categorical_columns))],\n    )\n    cb_model.fit(\n        cb_x_train,\n        y_train,\n        eval_set=[(cb_x_valid, y_valid)], \n        verbose=0,\n    )\n\n    train_oof_preds = cb_model.predict_proba(cb_x_valid)[:,1]\n    test_oof_preds = cb_model.predict_proba(test_df[cb_features])[:,1]\n    cb_train_preds[test_index] = train_oof_preds\n    cb_test_preds += test_oof_preds / n_folds\n    \n    print(": CB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))\n    \n    ########## Ridge model ##########\n    ridge_model = RidgeClassifier(\n        random_state=random_state,\n    )\n    ridge_model.fit(\n        ridge_x_train,\n        y_train,\n    )\n\n    train_oof_preds = ridge_model.decision_function(ridge_x_valid)\n    test_oof_preds = ridge_model.decision_function(test_df[ridge_features])\n    ridge_train_preds[test_index] = train_oof_preds\n    ridge_test_preds += test_oof_preds / n_folds\n    \n    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))\n    print("")\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\nprint("--> Overall metrics")\nprint(": XGB - ROC AUC Score = {}".format(\n    roc_auc_score(y, xgb_train_preds, average="micro")\n))\nprint(": LGB - ROC AUC Score = {}".format(\n    roc_auc_score(y, lgbm_train_preds, average="micro")\n))\nprint(": CB - ROC AUC Score = {}".format(\n    roc_auc_score(y, cb_train_preds, average="micro")\n))\nprint(": Ridge - ROC AUC Score = {}".format(\n    roc_auc_score(y, ridge_train_preds, average="micro")\n))\n')


# <a id='4.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">4.2 Blended classification</p>

# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\ny_train_preds = (\n    0.3 * xgb_train_preds +\n    0.4 * lgbm_train_preds +\n    0.3 * cb_train_preds\n)\n\nprint(": Essemble train test - ROC AUC Score = {}".format(\n    roc_auc_score(y, y_train_preds, average="micro")\n))\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "\ny_test_preds = (\n    0.3 * xgb_test_preds +\n    0.4 * lgbm_test_preds +\n    0.3 * cb_test_preds\n)\n\nsub_df['target'] = y_test_preds\nsub_df.to_csv('submission_base_essemble.csv',index=False)\n")


# <a id='4.3'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">4.3 L2 classification</p>
# 
# Predict the target over the L1 (level 1) predicted probabilities.

# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\nfrom scipy.special import expit\nfrom sklearn.calibration import CalibratedClassifierCV\n\nrandom_state = meta_random_seed\nn_folds = 10\nk_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)\n\nl1_train = pd.DataFrame(data={\n    "xgb": xgb_train_preds.tolist(),\n    "lgbm": lgbm_train_preds.tolist(),\n    "cb": cb_train_preds.tolist(),\n    "ridge": ridge_train_preds.tolist(),\n    "target": y.tolist()\n})\nl1_test = pd.DataFrame(data={\n    "xgb": xgb_test_preds.tolist(),\n    "lgbm": lgbm_test_preds.tolist(),\n    "cb": cb_test_preds.tolist(),\n    "ridge": ridge_test_preds.tolist(),    \n})\n\ntrain_preds = np.zeros(len(l1_train.index), )\ntest_preds = np.zeros(len(l1_test.index), )\nfeatures = ["xgb", "lgbm", "cb", "ridge"]\n\nfor fold, (train_index, test_index) in enumerate(k_fold.split(l1_train, y)):\n    print("--> Fold {}".format(fold + 1))\n    y_train = y.iloc[train_index]\n    y_valid = y.iloc[test_index]\n\n    x_train = pd.DataFrame(l1_train[features].iloc[train_index])\n    x_valid = pd.DataFrame(l1_train[features].iloc[test_index])\n    \n    model = CalibratedClassifierCV(\n        RidgeClassifier(random_state=random_state), \n        cv=3\n    )\n    model.fit(\n        x_train,\n        y_train,\n    )\n\n    train_oof_preds = model.predict_proba(x_valid)[:,-1]\n    test_oof_preds = model.predict_proba(l1_test[features])[:,-1]\n    train_preds[test_index] = train_oof_preds\n    test_preds += test_oof_preds / n_folds\n    \n    print(": ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))\n    print("")\n    \nprint("--> Overall metrics")\nprint(": ROC AUC Score = {}".format(roc_auc_score(y, train_preds, average="micro")))\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '\nsub_df["target"] = test_preds.tolist()\nsub_df.to_csv("submission_base_l2_classifier.csv", index=False)\n')


# <a id='5'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center">5. Models Optimization with Optuna ⛷</p>

# In[ ]:


def objective(trial, X=train_df[xgb_features], y=y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=meta_random_seed)


    lgb_params={
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
        'max_depth': trial.suggest_int('max_depth', 6, 200),
        'num_leaves': trial.suggest_int('num_leaves', 31, 120),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'random_state': meta_random_seed,
        'metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 6, 300000),
        'n_jobs': 12,
        'cat_feature': [x for x in range(len(categorical_columns))],
        'bagging_seed': 2021,
        'feature_fraction_seed': 2021,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 500),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.3, 0.9),
        'max_bin': trial.suggest_int('max_bin', 128, 1024),
        'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 350),
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 250),
        'cat_l2': trial.suggest_int('cat_l2', 1, 20)
    }

    lgb = LGBMClassifier(
        **lgb_params
    )
    lgb.fit(
        X_train,
        y_train,
        eval_set=(X_test,y_test),
        eval_metric='auc',
        early_stopping_rounds=100,
        verbose=False
    )
    predictions=lgb.predict_proba(X_test)[:,1]

    return roc_auc_score(y_test,predictions)


# In[ ]:


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, timeout=3600*7, n_trials=15)


# In[ ]:


# study.best_params


# In[ ]:


# Thanks to @gaetanlopez for the optimization result
lgbm_params={
    'learning_rate': 0.00605886703283976,
    'max_depth': 42,
    'num_leaves': 108,
    'reg_alpha': 0.9140720355379223,
    'reg_lambda': 9.97396811596188,
    'colsample_bytree': 0.2629101393563821,
    'min_child_samples': 61,
    'subsample_freq': 2,
    'subsample': 0.8329687190743886,
    'max_bin': 899,
    'min_data_per_group': 73,
    'cat_smooth': 21,
    'cat_l2': 11,
    'random_state': 2021,
    'metric': 'auc',
    'n_estimators': 20000,
    'n_jobs': -1,
    'bagging_seed': 2021,
    'feature_fraction_seed': 2021
}


# <a id='5.1'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">5.1 L1 Optimized Classification</p>

# In[ ]:


from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV

random_state = meta_random_seed
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

y = train_df["target"]

xgb_train_preds = np.zeros(len(train_df.index), )
xgb_test_preds = np.zeros(len(test_df.index), )
xgb_features = xgb_cat_features + list(numerical_columns)

lgbm_train_preds = np.zeros(len(train_df.index), )
lgbm_test_preds = np.zeros(len(test_df.index), )
lgbm_features = lgb_cat_features + list(numerical_columns)

cb_train_preds = np.zeros(len(train_df.index), )
cb_test_preds = np.zeros(len(test_df.index), )
cb_features = cb_cat_features + list(numerical_columns)

ridge_train_preds = np.zeros(len(train_df.index), )
ridge_test_preds = np.zeros(len(test_df.index), )
ridge_features = ridge_cat_features + list(numerical_columns)


# In[ ]:


for fold, (train_index, test_index) in enumerate(k_fold.split(train_df, y)):
    print("--> Fold {}".format(fold + 1))
    y_train = y.iloc[train_index]
    y_valid = y.iloc[test_index]

    ########## Generate train and valid sets ##########
    xgb_x_train = pd.DataFrame(train_df[xgb_features].iloc[train_index])
    xgb_x_valid = pd.DataFrame(train_df[xgb_features].iloc[test_index])

    lgbm_x_train = pd.DataFrame(train_df[lgbm_features].iloc[train_index])
    lgbm_x_valid = pd.DataFrame(train_df[lgbm_features].iloc[test_index])
    
    cb_x_train = pd.DataFrame(train_df[cb_features].iloc[train_index])
    cb_x_valid = pd.DataFrame(train_df[cb_features].iloc[test_index])

    ridge_x_train = pd.DataFrame(train_df[ridge_features].iloc[train_index])
    ridge_x_valid = pd.DataFrame(train_df[ridge_features].iloc[test_index])

    ########## XGBoost model ##########
    xgb_model = XGBClassifier(
        seed=random_state,
        n_estimators=10000,
        verbosity=1,
        eval_metric="auc",
        tree_method="gpu_hist",
        gpu_id=0,
        alpha=7.105034571323,
        colsample_bytree=0.25749283463,
        gamma=0.5003291821,
        reg_lambda=0.969826765347235612,
        learning_rate=0.009823136778823764,
        max_bin=338,
        max_depth=8,
        min_child_weight=2.2834723630466,
        subsample=0.6200435155855,
    )
    xgb_model.fit(
        xgb_x_train,
        y_train,
        eval_set=[(xgb_x_valid, y_valid)], 
        verbose=0,
        early_stopping_rounds=200
    )

    train_oof_preds = xgb_model.predict_proba(xgb_x_valid)[:,1]
    test_oof_preds = xgb_model.predict_proba(test_df[xgb_features])[:,1]
    xgb_train_preds[test_index] = train_oof_preds
    xgb_test_preds += test_oof_preds / n_folds
    
    print(": XGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    ########## LGBM model ##########
    lgbm_model = LGBMClassifier(
        cat_feature=[x for x in range(len(categorical_columns))],
        random_state=random_state,
        cat_l2=26.00385242730252,
        cat_smooth=89.2699690675538,
        colsample_bytree=0.2557260109926193,
        early_stopping_round=200,
        learning_rate=0.00605886703283976,
        max_bin=899,
        max_depth=42,
        metric="auc",
        min_child_samples=292,
        min_data_per_group=177,
        n_estimators=1600000,
        n_jobs=12,
        num_leaves=108,
        reg_alpha=0.9140720355379223,
        reg_lambda=5.643115293892745,
        subsample=0.919878341796,
        subsample_freq=1,
        verbose=-1,
    )
    lgbm_model.fit(
        lgbm_x_train,
        y_train,
        eval_set=[(lgbm_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = lgbm_model.predict_proba(lgbm_x_valid)[:,1]
    test_oof_preds = lgbm_model.predict_proba(test_df[lgbm_features])[:,1]
    lgbm_train_preds[test_index] = train_oof_preds
    lgbm_test_preds += test_oof_preds / n_folds
    
    print(": LGB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))

    ########## CatBoost model ##########
    cb_model = CatBoostClassifier(
        verbose=0,
        eval_metric="AUC",
        loss_function="Logloss",
        random_state=random_state,
        num_boost_round=20000,
        od_type="Iter",
        od_wait=200,
        task_type="GPU",
        devices="0",
        cat_features=[x for x in range(len(categorical_columns))],
        bagging_temperature=1.290192494969795,
        grow_policy="Depthwise",
        l2_leaf_reg=9.799870133539244,
        learning_rate=0.02017982653902465,
        max_depth=8,
        min_data_in_leaf=1,
        penalties_coefficient=2.096787602734,
    )
    cb_model.fit(
        cb_x_train,
        y_train,
        eval_set=[(cb_x_valid, y_valid)], 
        verbose=0,
    )

    train_oof_preds = cb_model.predict_proba(cb_x_valid)[:,1]
    test_oof_preds = cb_model.predict_proba(test_df[cb_features])[:,1]
    cb_train_preds[test_index] = train_oof_preds
    cb_test_preds += test_oof_preds / n_folds
    
    print(": CB - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    
    ########## Ridge model ##########
    ridge_model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state),
        cv=3,
    )
    ridge_model.fit(
        ridge_x_train,
        y_train,
    )

    train_oof_preds = ridge_model.predict_proba(ridge_x_valid)[:,-1]
    test_oof_preds = ridge_model.predict_proba(test_df[ridge_features])[:,-1]
    ridge_train_preds[test_index] = train_oof_preds
    ridge_test_preds += test_oof_preds / n_folds
    
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")


# <a id='5.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">5.2 L2 Optimized Classification</p>

# In[ ]:


from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV

random_state = meta_random_seed
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

l1_train = pd.DataFrame(data={
    "xgb": xgb_train_preds.tolist(),
    "lgbm": lgbm_train_preds.tolist(),
    "cb": cb_train_preds.tolist(),
    "ridge": ridge_train_preds.tolist(),
    "target": y.tolist()
})
l1_test = pd.DataFrame(data={
    "xgb": xgb_test_preds.tolist(),
    "lgbm": lgbm_test_preds.tolist(),
    "cb": cb_test_preds.tolist(),
    "ridge": ridge_test_preds.tolist(),    
})

l2_ridge_train_preds = np.zeros(len(l1_train.index), )
l2_ridge_test_preds = np.zeros(len(l1_test.index), )

l2_lgbm_train_preds = np.zeros(len(l1_train.index), )
l2_lgbm_test_preds = np.zeros(len(l1_test.index), )

features = ["xgb", "lgbm", "cb", "ridge"]

for fold, (train_index, test_index) in enumerate(k_fold.split(l1_train, y)):
    print("--> Fold {}".format(fold + 1))
    y_train = y.iloc[train_index]
    y_valid = y.iloc[test_index]

    x_train = pd.DataFrame(l1_train[features].iloc[train_index])
    x_valid = pd.DataFrame(l1_train[features].iloc[test_index])
    
    model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state), 
        cv=3
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,-1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,-1]
    l2_ridge_train_preds[test_index] = train_oof_preds
    l2_ridge_test_preds += test_oof_preds / n_folds
    
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
    model = LGBMClassifier(
        random_state=meta_random_seed
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,1]
    l2_lgbm_train_preds[test_index] = train_oof_preds
    l2_lgbm_test_preds += test_oof_preds / n_folds
    
    print(": LGBM - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y, l2_ridge_train_preds, average="micro")))
print(": LGBM - ROC AUC Score = {}".format(roc_auc_score(y, l2_lgbm_train_preds, average="micro")))


# In[ ]:


l1_train.head()


# ### Submission

# In[ ]:


sub_df["target"] = l2_lgbm_test_preds.tolist()
sub_df.to_csv("submission_optimized_l2_lgbm_classifier.csv", index=False)

sub_df["target"] = l2_ridge_test_preds.tolist()
sub_df.to_csv("submission_optimized_l2_ridge_classifier.csv", index=False)


# <a id='6'></a>
# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:150%; text-align:center">6. H2O AutoML</p>

# In[ ]:


import h2o
from h2o.automl import H2OAutoML

h2o.init(
    max_mem_size=14,
    nthreads=12,
)


# In[ ]:


x = train_df[cb_features]

hf = h2o.H2OFrame(pd.concat([x, y], axis=1))
x_test_hf = h2o.H2OFrame(test_df[cb_features])

hf.head()


# In[ ]:


predictors = hf.columns
label = "target"
predictors.remove(label)

hf[label] = hf[label].asfactor()

train_hf, valid_hf  = hf.split_frame(ratios=[.8], seed=meta_random_seed)


# In[ ]:


train_hf.describe()


# In[ ]:


aml = H2OAutoML(
    max_runtime_secs=3600, 
    seed=meta_random_seed,
    exclude_algos = ["DeepLearning", "DRF"]
)


# In[ ]:


aml.train(
    x=predictors,
    y=label, 
    training_frame=train_hf,
    validation_frame=valid_hf,
)


# In[ ]:


# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=10)


# In[ ]:


aml_leader_test_preds = aml.predict(x_test_hf).as_data_frame()['p1']


# In[ ]:


sub_df['target'] = list(aml_leader_test_preds)
sub_df.to_csv("submission_aml_leader_30min.csv", index=False)


# ### Predict with the TOP 5 models

# In[ ]:


# Get model ids for all models in the AutoML Leaderboard
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])


# In[ ]:


tmp_model = h2o.get_model(model_ids[2])
tmp_model.auc(valid=True)


# In[ ]:


aml_train_preds = {}
aml_test_preds = {}

for model_id in model_ids[:5]:
    tmp_model = h2o.get_model(model_id)
    
    aml_train_preds[model_id] = list(tmp_model.predict(hf).as_data_frame()['p1'])
    aml_test_preds[model_id] = list(tmp_model.predict(x_test_hf).as_data_frame()['p1'])

aml_train_preds['target'] = list(y)

aml_train_preds_df = pd.DataFrame(data = aml_train_preds)
aml_test_preds_df = pd.DataFrame(data = aml_test_preds)


# In[ ]:


aml_train_preds_df.head()


# In[ ]:


aml_train_preds_df.shape, aml_test_preds_df.shape


# <a id='6.2'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">6.2 L2 AutoML Classification</p>

# In[ ]:


random_state = meta_random_seed
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

aml_l2_ridge_train_preds = np.zeros(len(aml_train_preds_df.index), )
aml_l2_ridge_test_preds = np.zeros(len(aml_test_preds_df.index), )

aml_l2_lgbm_train_preds = np.zeros(len(aml_train_preds_df.index), )
aml_l2_lgbm_test_preds = np.zeros(len(aml_test_preds_df.index), )

features = aml_test_preds_df.columns

for fold, (train_index, test_index) in enumerate(k_fold.split(aml_train_preds_df, y)):
    print("--> Fold {}".format(fold + 1))
    y_train = y.iloc[train_index]
    y_valid = y.iloc[test_index]

    x_train = pd.DataFrame(aml_train_preds_df[features].iloc[train_index])
    x_valid = pd.DataFrame(aml_train_preds_df[features].iloc[test_index])
    
    model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state), 
        cv=3
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,-1]
    test_oof_preds = model.predict_proba(aml_test_preds_df[features])[:,-1]
    aml_l2_ridge_train_preds[test_index] = train_oof_preds
    aml_l2_ridge_test_preds += test_oof_preds / n_folds
    
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
    model = LGBMClassifier(
        random_state=meta_random_seed
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,1]
    test_oof_preds = model.predict_proba(aml_test_preds_df[features])[:,1]
    aml_l2_lgbm_train_preds[test_index] = train_oof_preds
    aml_l2_lgbm_test_preds += test_oof_preds / n_folds
    
    print(": LGBM - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y, aml_l2_ridge_train_preds, average="micro")))
print(": LGBM - ROC AUC Score = {}".format(roc_auc_score(y, aml_l2_lgbm_train_preds, average="micro")))


# In[ ]:


sub_df["target"] = aml_l2_lgbm_test_preds.tolist()
sub_df.to_csv("submission_aml_l2_lgbm_classifier_top5.csv", index=False)

sub_df["target"] = aml_l2_ridge_test_preds.tolist()
sub_df.to_csv("submission_aml_l2_ridge_classifier_top5.csv", index=False)


# <a id='6.3'></a>
# ## <p style="background-color:skyblue; font-family:newtimeroman; font-size:120%; text-align:center">6.3 L2 Optimized + AutoML leader Classification</p>

# In[ ]:


aml_leader_test_preds = aml.predict(x_test_hf).as_data_frame()['p1']
aml_leader_train_preds = aml.predict(hf).as_data_frame()['p1']


# In[ ]:


from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV

random_state = meta_random_seed
n_folds = 10
k_fold = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

l1_train = pd.DataFrame(data={
    "xgb": xgb_train_preds.tolist(),
    "lgbm": lgbm_train_preds.tolist(),
    "cb": cb_train_preds.tolist(),
    "ridge": ridge_train_preds.tolist(),
    "aml": list(aml_leader_train_preds),
    "target": y.tolist()
})
l1_test = pd.DataFrame(data={
    "xgb": xgb_test_preds.tolist(),
    "lgbm": lgbm_test_preds.tolist(),
    "cb": cb_test_preds.tolist(),
    "ridge": ridge_test_preds.tolist(),
    "aml": list(aml_leader_test_preds)
})

l2_ridge_train_preds = np.zeros(len(l1_train.index), )
l2_ridge_test_preds = np.zeros(len(l1_test.index), )

l2_lgbm_train_preds = np.zeros(len(l1_train.index), )
l2_lgbm_test_preds = np.zeros(len(l1_test.index), )

features = ["xgb", "lgbm", "cb", "ridge", "aml"]

for fold, (train_index, test_index) in enumerate(k_fold.split(l1_train, y)):
    print("--> Fold {}".format(fold + 1))
    y_train = y.iloc[train_index]
    y_valid = y.iloc[test_index]

    x_train = pd.DataFrame(l1_train[features].iloc[train_index])
    x_valid = pd.DataFrame(l1_train[features].iloc[test_index])
    
    model = CalibratedClassifierCV(
        RidgeClassifier(random_state=random_state), 
        cv=3
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,-1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,-1]
    l2_ridge_train_preds[test_index] = train_oof_preds
    l2_ridge_test_preds += test_oof_preds / n_folds
    
    print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
    model = LGBMClassifier(
        random_state=meta_random_seed
    )
    model.fit(
        x_train,
        y_train,
    )

    train_oof_preds = model.predict_proba(x_valid)[:,1]
    test_oof_preds = model.predict_proba(l1_test[features])[:,1]
    l2_lgbm_train_preds[test_index] = train_oof_preds
    l2_lgbm_test_preds += test_oof_preds / n_folds
    
    print(": LGBM - ROC AUC Score = {}".format(roc_auc_score(y_valid, train_oof_preds, average="micro")))
    print("")
    
print("--> Overall metrics")
print(": Ridge - ROC AUC Score = {}".format(roc_auc_score(y, l2_ridge_train_preds, average="micro")))
print(": LGBM - ROC AUC Score = {}".format(roc_auc_score(y, l2_lgbm_train_preds, average="micro")))


# ### Submission

# In[ ]:


sub_df["target"] = l2_lgbm_test_preds.tolist()
sub_df.to_csv("submission_optimized_aml_l2_lgbm_classifier.csv", index=False)

sub_df["target"] = l2_ridge_test_preds.tolist()
sub_df.to_csv("submission_optimized_aml_l2_ridge_classifier.csv", index=False)

