#!/usr/bin/env python
# coding: utf-8

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300"> 🩺 Notebook At a Glance</p>

# ![image.png](attachment:5fd8ab0b-c50d-4ddf-97fb-f31aeb12cdac.png)

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import gc
import re as re
from collections import Counter

from tqdm.auto import tqdm
import math
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

import time
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
get_ipython().run_line_magic('matplotlib', 'inline')
tqdm.pandas()

rc = {
    "axes.facecolor": "#FFF9ED",
    "figure.facecolor": "#FFF9ED",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL


# In[3]:


# check the path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


train = pd.read_csv('/kaggle/input/playground-series-s3e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e12/test.csv')
original = pd.read_csv('/kaggle/input/kidney-stone-prediction-based-on-urine-analysis/kindey stone urine analysis.csv')


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 data description:</font></h3>
# 
# * The six physical characteristics of the urine are: 
#     * (1) specific gravity, the density of the urine relative to water; 
#     * (2) pH, the negative logarithm of the hydrogen ion; 
#     * (3) osmolarity (mOsm), a unit used in biology and medicine but not in physical chemistry. Osmolarity is proportional to the concentration of
# molecules in solution; 
#     * (4) conductivity (mMho milliMho). One Mho is one
# reciprocal Ohm. Conductivity is proportional to the concentration of charged
# ions in solution; 
#     * (5) urea concentration in millimoles per litre; and 
#     * (6) calcium concentration (CALC) in millimolesllitre.

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">📊 EDA</p>

# In[5]:


# summary table function
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


# In[6]:


summary(train)


# In[7]:


summary(test)


# In[8]:


summary(original)


# #### > 📊 summary table 
# - summary table shows missing value (%), unique value, minimum and maximum value of each variables.
# - Also, you can check each columns' first 3 values, which could help you to get grasp of whole dataset.

# In[9]:


fig2 = px.pie(train, names='target', 
             height=400, width=600, 
             hole=0.7, 
             title='target class Overview',
                   color_discrete_sequence=['#4c78a8', '#72b7b2'])
fig2.update_traces(hovertemplate=None, textposition='outside', textinfo='percent+label', rotation=0)
fig2.update_layout(margin=dict(t=100, b=30, l=0, r=0), showlegend=False,
                        plot_bgcolor='#fafafa', paper_bgcolor='#fafafa',
                        title_font=dict(size=20, color='#555', family="Lato, sans-serif"),
                        font=dict(size=17, color='#8a8d93'),
                        hoverlabel=dict(bgcolor="#444", font_size=13, font_family="Lato, sans-serif"))
fig2.show()


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Insights:</font></h3>
# 
# * This is a balanced classification dataset where 44.4% are positive(1) patient and 55.6% are negative(0) one.

# In[10]:


num_cols = test.select_dtypes(include=['float64','int64']).columns.tolist()
num_cols.remove('id')


# In[11]:


get_ipython().run_cell_magic('time', '', "figsize = (6*6, 20)\nfig = plt.figure(figsize=figsize)\nfor idx, col in enumerate(num_cols):\n    ax = plt.subplot(2,3, idx + 1)\n    sns.kdeplot(\n        data=train, hue='target', fill=True,\n        x=col, palette=['#9E3F00', 'red'], legend=False\n    )\n            \n    ax.set_ylabel(''); ax.spines['top'].set_visible(False), \n    ax.set_xlabel(''); ax.spines['right'].set_visible(False)\n    ax.set_title(f'{col}', loc='right', \n                 weight='bold', fontsize=20)\n\nfig.suptitle(f'Features vs Target\\n\\n\\n', ha='center',  fontweight='bold', fontsize=21)\nfig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=21, ncol=3)\nplt.tight_layout()\nplt.show()\n")


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Insights:</font></h3>
# 
# * The distribution is different obviously, which means these variables have predictive power in some sense.
# * Therefore, we are going to use all these variables without applying feature selection method.

# In[12]:


features = num_cols
n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}

columns = features
n_cols = 4
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    sns.kdeplot(
        train[column], label='Train',
        ax=ax[i], color='#9E3F00'
    )
    
    sns.kdeplot(
        test[column], label='Test',
        ax=ax[i], color='yellow'
    )
    
    
    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    
    # remove axes to show only one at the end
    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]
        plot_ax.legend().remove()
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')
    
fig.suptitle(f'Numerical Feature Distributions\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=25, ncol=3)
plt.tight_layout()


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Insights:</font></h3>
# 
# * The distribution between train and test dataset is almost same.

# In[13]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    corr = df.corr()
    fig, axes = plt.subplots(figsize=(10, 5))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='YlOrRd', annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(train[num_cols], 'Train Dataset Correlation')
plot_correlation_heatmap(test[num_cols], 'Test Dataset Correlation')


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# * Some variables shows high correlations. Ex, urea and osmo, cond and osmo.
# * We are going to use tree-based model as our baseline. As tree-based models have an innate feature of being robust to correlated features, we are not going to consider removing highly correlated variables in the first place.

# #### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">🛠 Modeling</p>

# <div class="alert alert-block alert-success" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Modeling overview:</u></b><br>
#     
# * <i> build baseline model without hyperparameter tuning</i><br>
# * <i> 10-fold cross validation methods are used for baseline modeling.</i><br>
# * <i> metric is AUC</i><br>
#   
# </div>

# In[14]:


TARGET = 'target'
FEATURES = num_cols


# In[15]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(train[FEATURES])
train[FEATURES] = scaler.transform(train[FEATURES])
test[FEATURES] = scaler.transform(test[FEATURES])


# In[16]:


def f_importance_plot(f_imp):
    fig = plt.figure(figsize=(12, 0.20*len(f_imp)))
    plt.title('Feature importances', size=16, y=1.05, 
              fontweight='bold', color='#444444')
    a = sns.barplot(data=f_imp, x='avg_imp', y='feature', 
                    palette='hot', linestyle="-", 
                    linewidth=0.5, edgecolor="black")
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks(size=11, color='#444444')
    
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    for j in ['left']:
        a.spines[j].set_linewidth(0.5)
    plt.tight_layout()
    plt.show()
    
def show_confusion_roc(oof: list) -> None:
    """Draws a confusion matrix and roc_curve with AUC score.
        
        Args:
            oof: predictions for each fold stacked. (list of tuples)
        
        Returns:
            None
    """
    
    f, ax = plt.subplots(1, 2, figsize=(13.3, 4))
    df = pd.DataFrame(np.concatenate(oof), columns=['id', 'preds', 'target']).set_index('id')
    df.index = df.index.astype(int)
    cm = confusion_matrix(df.target, df.preds.ge(0.5).astype(int))
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap='YlOrRd', ax=ax[0])
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, color='#458B00', ax=ax[1])
    plt.tight_layout();
    
def get_mean_auc(oof: np.array):
    """oof: ['val_idx', 'preds', 'target']"""
    oof = pd.DataFrame(np.concatenate(oof), columns=['id', 'preds', 'target']).set_index('id')
    oof.index = oof.index.astype(int)
    mean_val_auc = roc_auc_score(oof.target, oof.preds)
    return mean_val_auc


# In[17]:


#reference: https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission
from xgboost import XGBClassifier
FOLDS = 10
SEED = 1004
xgb_models = []
xgb_oof = []
predictions = np.zeros(len(test))
f_imp = []

counter = 1
X = train.drop(columns=['target'])
y = train.target
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    if (fold + 1)%5 == 0 or (fold + 1) == 1:
        print(f'{"#"*24} Training FOLD {fold+1} {"#"*24}')
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    watchlist = [(X_train, y_train), (X_valid, y_valid)]

    # XGboost model and fit
    model = XGBClassifier(n_estimators=1000, n_jobs=-1, max_depth=4, eta=0.2, colsample_bytree=0.67)
    model.fit(X_train, y_train, eval_set=watchlist, early_stopping_rounds=300, verbose=0)

    val_preds = model.predict_proba(X_valid)[:, 1]
    val_score = roc_auc_score(y_valid, val_preds)
    best_iter = model.best_iteration

    idx_pred_target = np.vstack([val_idx,  val_preds, y_valid]).T # shape(len(val_idx), 3)
    f_imp.append({i: j for i in model.feature_names_in_ for j in model.feature_importances_ })
    print(f'{" "*20} auc:{blu}{val_score:.5f}{res} {" "*6} best iteration  :{blu}{best_iter}{res}')

    xgb_oof.append(idx_pred_target)
    xgb_models.append(model)
#     test_preds = model.predict_proba(test)[:,1] / FOLDS
#     predictions += test_preds
    if val_score > 0.80:
        test_preds = model.predict_proba(test)[:,1] 
        predictions += test_preds
        counter += 1

predictions /= counter
mean_val_auc = get_mean_auc(xgb_oof)
print('*'*45)
print(f'{red}Mean{res} AUC: {red}{mean_val_auc:.5f}{res}')


# In[18]:


#reference: https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission
show_confusion_roc(xgb_oof)
f_imp_df = pd.DataFrame(f_imp).mean().reset_index()
f_imp_df.columns = ['feature', 'avg_imp']
f_importance_plot(f_imp_df)


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Conclusion:</font></h3>
# 
# * According to it's AUC from cross-validation, we could see that these dataset have strong preditive power.
# 
# * I skipped detailed feature engineering and hyper-parameter tuning, which could enhance model performance.

# #### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">Acknowledgement</p>

# Modeling and evaluation code: https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e7-2023-eda-and-submission
# 
# @jcaliz for .css and plotting ideas.
