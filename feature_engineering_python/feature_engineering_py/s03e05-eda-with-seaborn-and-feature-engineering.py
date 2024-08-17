#!/usr/bin/env python
# coding: utf-8

# # Season 2023, Episode 5: Wine Quality
# 
# This is a starting version of the notebook. It currently only contains some preliminary EDA visualizations in a clean plotting style.
# 
# ## Table of Contents
# * 1. [Data](#1)
# * 2. [Exploratory Data Analysis](#2)
#     * a. [Target Variable](#2a)
#     * b. [Distribution of Features](#2b)
#     * c. [Correlation among Variables](#2c)
#     * d. [Relationship between Features and Wine Quality](#2d)
#     * e. [Relationship among Original Features](#2e)
# * 3. [Feature Engineering](#3)
#     * a. [Acidic Features](#3a)
#     * b. [Sulfur Features](#3b)
#     * c. [Alcohol Features](#3c)
# * 4. [Modelling](#4)
#     * a. [Logistic Regression](#4a)
#         * [Tuning](#4a1)
#         * [Results](#4a2)

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().system('pip install seaborn==0.12.2')
import seaborn as sns

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')


# <a id="1"></a>
# # 1. Data

# In[2]:


train = pd.read_csv("/kaggle/input/playground-series-s3e5/train.csv").assign(sample = 'train')
test = pd.read_csv("/kaggle/input/playground-series-s3e5/test.csv").assign(sample = 'test')
df = pd.concat([train, test], ignore_index=True)

original_features = test.drop(columns=['Id', 'sample']).columns

df.head()


# In[3]:


print(train.shape)
print(test.shape)


# In[4]:


print(df.isnull().sum())


# # 2. Exploratory Data Analysis
# <a id="2"></a>

# In[5]:


main_color, first_color, second_color = '#800020', "#15616D", "#FF7D00"
#4C5760 Black coral
#93A8AC
#A59E8C
#D7CEB2


# ## 2a. Target Variable: quality
# <a id="2a"></a>

# In[6]:


fig = plt.figure(figsize=(8,4))
ax = sns.countplot(data=df.query("sample == 'train'"), x='quality', color=main_color, edgecolor='black')
ax.set_facecolor('#F1ECCE')
fig.set_facecolor('#F1ECCE')
ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
ax.set_axisbelow(True)

ax.set_xlabel('Quality')
ax.set_ylabel('Count')
ax.set_ylim(0,950)


counts = train.groupby('quality')['Id'].count().divide(len(train))
for i, p in enumerate(ax.patches):
    x,_ = p.get_xy()
    y = p.get_height()
    w = p.get_width()
    ax.text(x+w/2, y+10, s=f'{round(counts.iloc[i]*100,1)}%', ha='center')


# In this training set, most wine quality is either rated at 5 or 6 (around 78%). 

# ## 2b. Distribution of Features
# <a id="2b"></a>

# In[7]:


fig = plt.figure(figsize=(22,16))
fig.set_facecolor('#F1ECCE')
for i, col in enumerate(original_features):
    fig.add_subplot(3,4, i+1)
    ax = sns.histplot(data=df.query("sample == 'train'"), x=col, bins=20, color=main_color, edgecolor='black', alpha=1)
    ax.set_facecolor('#F1ECCE')
    ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
    ax.set_axisbelow(True)
    ax.set_xlabel('')
    ax.set_title(col, weight='bold')


# ## 2c. Correlation among Variables
# <a id="2c"></a>

# In[8]:


df_corr = df.drop(columns='Id').corr()
mask = np.triu(np.ones_like(df_corr, dtype=bool))[1:,:-1]

corr = df_corr.iloc[1:,:-1].copy()

# color scale
cmap = sns.diverging_palette(10,10,s=100,l=25, as_cmap=True)

fig = plt.figure(figsize=(14,12))
fig.set_facecolor('#F1ECCE')

ax = sns.heatmap(corr, annot=True, fmt=".2f", mask=mask, square=True,cmap=cmap, vmin=-1, vmax=1, cbar_kws={"shrink": .8}, linewidths=5, linecolor="#F1ECCE")
ax.set_facecolor('#F1ECCE')




# ## 2d. Relationship between Features and Wine Quality
# <a id="2d"></a>

# In[9]:


fig = plt.figure(figsize=(22,16))
fig.set_facecolor('#F1ECCE')
for i, col in enumerate(original_features):
    fig.add_subplot(3,4, i+1)
    ax = sns.boxenplot(data=df, x='quality', y=col, color=main_color,
                      line_kws={'color':'white'})
    ax.set_facecolor('#F1ECCE')
    ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
    ax.set_axisbelow(True)
    ax.set_xlabel('Quality')
    ax.set_title(col, weight='bold')


# We see some relationships between the quality of the wine and individual features of the wine:
# * Higher alcohol value tends to indicate a *higher* wine quality
# * Higher sulphates value tends to indicate a *higher* wine quality
# * Higher density value tends to indicate a *lower* wine quality
# * Higher chlorides value tends to indicate a *lower* wine quality

# ## 2e. Relationship among Original Features
# <a id="2e"></a>

# In[10]:


g = sns.pairplot(data=train, hue='quality', corner=True, palette='Reds' , diag_kind='kde',
             x_vars=original_features, y_vars=original_features,
                plot_kws = {'edgecolor':'black'})
g.fig.set_facecolor('#F1ECCE')

sns.move_legend(g, loc='upper center', fontsize=20, markerscale=2, edgecolor='black', title='Quality', title_fontsize=24, ncol=6)

# Change edgecolor for legend points to black
for ha in g.legend.legendHandles:
    ha.set_edgecolor('black')

# Set facecolor for each individual plot
for ax in g.axes.flat:
    if ax != None:
        ax.set_facecolor('#F1ECCE')
        if ax.get_xlabel() != None:
            ax.set_xlabel(ax.get_xlabel(), weight='bold')
        if ax.get_ylabel() != None:
            ax.set_ylabel(ax.get_ylabel(), weight='bold')


# # 3. Feature Engineering
# <a id="3"></a>
# Since there are no missing values in this dataset I will just focus on creating/modifying features.

# ## 3a. Acidic Features
# <a id="3a"></a>
# 
# Total acidity tells us the concentration of acids present in wine, whereas the pH level tells us how intense those acids taste. A wine with too much acidity will taste excessively sour and sharp. A wine with too little acidity will taste flabby and flat, with less defined flavors.
# 
# * **fixed acidity**: The predominant fixed acids found in wines are tartaric, malic, citric, and succinic. Their respective levels found in wine can vary greatly but in general one would expect to see:
#     * 1 to 4 g/L tartaric acid,
#     * 0 to 8 g/L malic acid,
#     * 0 to 0.5 g/L citric acid, and
#     * 0.5 to 2 g/L succinic acid.
# * **volatile acidity**: Represents the fraction of acidic substances which can be freed from wine belonging to the acetic series and perceivable both to smell and taste. The two main volatile acids are:
#     * acetic acid
#     * ethyl acetate
# * **citric acid**: Citric acid is often added to wines to increase acidity, complementing a specific flavor. Typically 0-0.5 g/l is in red wines
# * **pH**: The pH of a wine is a measure of the strength and concentration of the dissociated acids present
# * **sulphates**: Sulphates help preserve wine and slow chemical reactions, which cause a wine to go bad
# 
# Some notes about winemaking (attempt to derive new features through domain knowledge - I have very little on winemaking):
# * Wines with lower acidity need more sulfites than higher acidity wines. At pH 3.6 or greater, wines are less stable and need additional sulphates.

# In[11]:


df['citric per volatile'] = df['citric acid'] / df['volatile acidity']
df['citric per fixed'] = df['citric acid'] / df['fixed acidity']
df['volatile per fixed'] = df['volatile acidity'] / df['fixed acidity']

df['pH per sulphates'] = df['pH']/df['sulphates']
df['pH per chlorides'] = df['pH'] / df['chlorides']

df['fixed acidity per chlorides'] = df['fixed acidity'] / df['chlorides']

acid_engineer_features = ['citric per volatile', 'citric per fixed', 'volatile per fixed', 'pH per sulphates', 'pH per chlorides', 'fixed acidity per chlorides']


# In[12]:


fig = plt.figure(figsize=(16,12))
plt.subplots_adjust(hspace=0.35)

fig.set_facecolor('#F1ECCE')
for i, col in enumerate(acid_engineer_features):
    fig.add_subplot(2,3, i+1)
    ax = sns.boxenplot(data=df.query("sample == 'train'"), x='quality', y=col, color=main_color, 
                      line_kws={'color':'white'})
    ax.set_facecolor('#F1ECCE')
    ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
    ax.set_axisbelow(True)
    ax.set_xlabel('Quality')
    ax.set_title(col, weight='bold')


# ## 3b. Sulfur Features
# <a id="3b"></a>
# * Sulfur dioxide preserves wine and prevents oxidation and browning. It is usually added more to white than red wines as they have less of other antioxidants.
# * **total sulfur dioxide** = **free sulfur dioxide** + **bound sulfur dioxide** 
# * The amount of free sulfur dioxide to add to maintain the proper molecular level is dependent on the wine’s pH.
# 
# From (https://www.extension.iastate.edu/wine/total-sulfur-dioxide-why-it-matters-too/)

# In[13]:


df['bound sulfur dioxide'] = (df['total sulfur dioxide'] - df['free sulfur dioxide'])
df['sulfur ratio'] = df['free sulfur dioxide']/df['total sulfur dioxide']

sulfur_features = ['bound sulfur dioxide', 'sulfur ratio']


# In[14]:


fig = plt.figure(figsize=(16, 10))
plt.subplots_adjust(hspace=0.35)

fig.set_facecolor('#F1ECCE')
for i, col in enumerate(sulfur_features):
    fig.add_subplot(2,3, i+1)
    ax = sns.boxenplot(data=df.query("sample == 'train'"), x='quality', y=col, color=main_color, line_kws={'color':'white'})
    ax.set_facecolor('#F1ECCE')
    ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
    ax.set_axisbelow(True)
    ax.set_xlabel('Quality')
    ax.set_title(col, weight='bold')


# ## 3c. Alcohol Features
# <a id="3c"></a>

# In[15]:


df['alcohol per density'] = df['alcohol'] / df['density']
df['alcohol per pH'] = df['alcohol'] / df['pH']
df['alcohol per volatile acidity'] = df['alcohol'] / df['volatile acidity']
df['alcohol per fixed acidity'] = df['alcohol'] / df['volatile acidity']
df['alcohol per total sulfur dioxide'] = df['alcohol'] / df['total sulfur dioxide']
df['alcohol per residual sugar'] = df['alcohol'] / df['residual sugar']

df['sulphates per chlorides'] = df['sulphates'] / df['chlorides']
df['alcohol per sulphates'] = df['alcohol'] / df['sulphates']

alcohol_features = df.filter(like='alcohol per').columns


# In[16]:


fig = plt.figure(figsize=(16,20))
fig.set_facecolor('#F1ECCE')
for i, col in enumerate(alcohol_features):
    fig.add_subplot(3,3, i+1)
    ax = sns.boxenplot(data=df.query("sample == 'train'"), x='quality', y=col, color=main_color, line_kws={'color':'white'})
    ax.set_facecolor('#F1ECCE')
    ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
    ax.set_axisbelow(True)
    ax.set_xlabel('Quality')
    ax.set_title(col, weight='bold')


# # 4. Modelling
# <a id="4"></a>

# ## 4a. Logistic Regression
# <a id="4a"></a>

# In[17]:


target = 'quality'
features = ['alcohol', 'sulphates' , 'alcohol per sulphates', 'sulphates per chlorides', 'chlorides']


# In[18]:


X_train = df.query("sample == 'train'")[features]
y_train = df.query("sample == 'train'").quality
train = pd.concat([X_train, y_train], axis=1)
test = df.query("sample == 'test'")[features].copy()


# ### 4a-1. Tuning
# <a id="4a1"></a>

# In[19]:


def objective(trial):
    """Define the objective function"""

    params = {
        'C': trial.suggest_float('C', 0.001, 1000),
        'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag']),
        #'penalty': trial.suggest_categorical('penalty', [None, 'l1', 'l2', 'elasticnet']),
        'max_iter': trial.suggest_int('max_iter', 100, 10000),
        'multi_class': 'auto',
        'random_state': 13
    }

    cv = StratifiedKFold(10, shuffle=True, random_state=13)
    fold_scores = []
    for i, (train_idx,val_idx) in enumerate(cv.split(train[features],train[target])):
        X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, target]
        X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, target]
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        pred_val = model.predict(X_val)
        pred_test = model.predict(test[features])

        score = cohen_kappa_score(y_val,pred_val, weights='quadratic')
        fold_scores.append(score)
        
        if len(fold_scores) == 10:
            print(fold_scores)
        
    return np.mean(fold_scores)


# In[20]:


#study = optuna.create_study(direction='maximize', sampler = TPESampler())
#study.optimize(objective, n_trials=100)


# ### 4a-2. Results
# <a id="4a2"></a>

# | Feature Set | Tuned Hyperparameters | CV (10-fold) Score | LB Score |
# |---|---|---|---|
# | Original Features | {'C': 657.9216543990904, 'solver': 'newton-cg', 'max_iter': 4101} | 0.4558 | |
# | Alcohol | {'C': 257.4563826622824, 'solver': 'newton-cg', 'max_iter': 5033} | 0.3698 | |
# | Alcohol + Sulphates | {'C': 299.27463844709087, 'solver': 'lbfgs', 'max_iter': 6404} | 0.4789 | |
# | Alcohol + Sulphates + Alcohol/Sulphates | {'C': 92.72537339506918, 'solver': 'lbfgs', 'max_iter': 144} | 0.5018 | 0.50426 |

# In[21]:


g = sns.jointplot(data=df, x="alcohol", y="sulphates", hue='quality', palette='Reds', edgecolor='black', kind='scatter', )
g.ax_marg_x.set_facecolor('#F1ECCE')
g.ax_marg_y.set_facecolor('#F1ECCE')
g.fig.set_facecolor('#F1ECCE')
g.ax_joint.set_facecolor('#F1ECCE')
ax = plt.gca()
leg = ax.legend(title='Quality', facecolor='#F1ECCE', edgecolor='black')

# Change edgecolor for legend points to black
for ha in leg.legendHandles:
    ha.set_edgecolor('black')


# In[22]:


params = {'C': 92.72537339506918, 'solver': 'lbfgs', 'max_iter': 144}
clf = LogisticRegression(**params)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)

target_df = pd.concat([pd.DataFrame(clf.predict(test), columns=['predicted_quality']), train.quality], axis=1).melt()\
                .groupby('variable')['value']\
                .value_counts(normalize=True)\
                .mul(100)\
                .rename('Percent')\
                .reset_index()


# In[23]:


sns.set_palette(sns.color_palette([first_color, second_color]))
g = sns.catplot(data=target_df, x='value',y='Percent',hue='variable', kind='bar', edgecolor='black', facet_kws={'despine':False})
g.fig.set_facecolor('#F1ECCE')
for ax in g.axes.flat:
    ax.set_facecolor('#F1ECCE')
    ax.grid(axis='y', color='black', linestyle='--', alpha=0.5, zorder=3)
    ax.set_axisbelow(True)    
    ax.set_xlabel('Quality')

