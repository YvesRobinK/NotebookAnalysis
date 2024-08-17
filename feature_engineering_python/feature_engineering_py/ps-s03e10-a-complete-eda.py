#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
get_ipython().system('pip install feature-engine 2>/dev/null 1>&2')
get_ipython().system('pip install opentsne 2>/dev/null 1>&2')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# # PS S03E10: A Complete EDA ⭐️
# 
# This EDA gives useful insights when designig a machine learning pipeline for this 2-week competition
# 
# **Versions**
# - v1: Inital EDA
# - v2: Added feature importances
# - v3: Added pseudo-duplicates analysis.
# - v4: Added some post-processing techniques
# - v5: minor fix on a palette, added `is_generated` feature to increase CV.
# - v6: Added robustScaler + postprocessing based on ceil and floor.
# - v7: probability Calibration [see this post](https://www.kaggle.com/competitions/playground-series-s3e10/discussion/393861) by [@sergiosaharovskiy](https://www.kaggle.com/sergiosaharovskiy)
# - v8: Added some feature engineering inspired in this [post](https://www.kaggle.com/competitions/playground-series-s3e10/discussion/394564) by [@jimgruman](https://www.kaggle.com/jimgruman) oof: 0.30965 → 0.3081
# - v9: Optimized LGBM oof: 0.3081 → 0.3068
# - v10: Switch to RepeatedStratifiedKFold oof: 0.3068 → 0.03056. The reason for this switch is because there are several folds on which val is better than train. This issue reflects that some folds have easier examples in validation, therefore more models will compensate overfitting to easier examples.
# - V11: addeda TSNE plot
# 
# # Table of Content
# 
# 1. [The Data](#The-Data)
#     1. [What IS DM SNR CURVE:](#What-IS-DM-SNR-CURVE:)
# 1. [The Label](#The-Label)
# 1. [EDA](#EDA)
# 1. [The data Size](#The-data-Size)
# 1. [The Duplicates](#The-Duplicates)
# 1. [The Pseudo Duplicates](#The-Pseudo-Duplicates)
#     1. [Rounding](#Rounding)
#     1. [Using np.ceil](#Using-np.ceil)
#     1. [Using np.floor](#Using-np.floor)
# 1. [The TSNE](#The-TSNE)
# 1. [The Distributions](#The-Distributions)
# 1. [The Adversarial Validation](#The-Adversarial-Validation)
# 1. [The Correlations](#The-Correlations)
# 1. [The Feature Engineering](#The-Feature-Engineering)
# 1. [The Baseline](#The-Baseline)
# 1. [The Feature Importance](#The-Feature-Importance)
# 1. [The postprocessing](#The-postprocessing)
#     1. [Set predictions that are close to 1 or close to 0, to 1 and 0 respectively.](#Set-predictions-that-are-close-to-1-or-close-to-0,-to-1-and-0-respectively.)
#     1. [Find the best threshold and update each side of the plot](#Find-the-best-threshold-and-update-each-side-of-the-plot)
#     1. [Ceil and Floor](#Ceil-and-Floor)
# 1. [The Calibration](#The-Calibration)
# 1. [The Submission](#The-Submission)
# 1. [The Comparison OOF vs Test](#The-Comparison-OOF-vs-Test)

# # Library Import
# 
# 
# Some library import, the usual stuff

# In[2]:


from time import time
from datetime import timedelta
from colorama import Fore, Style
from lightgbm import early_stopping
from lightgbm import log_evaluation

import math
import matplotlib
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lightgbm as lgbm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import warnings
from cycler import cycler
from matplotlib.ticker import MaxNLocator
from feature_engine.selection import DropCorrelatedFeatures

palette = ['#3c3744', '#048BA8', '#EE6352', '#E1BB80', '#78BC61']
grey_palette = [
    '#8e8e93', '#636366', '#48484a', '#3a3a3c', '#2c2c2e', '#1c1c27'
]

bg_color = '#F6F5F5'
white_color = '#d1d1d6'

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": False,
    'grid.alpha':0.2,
    'figure.figsize': (16, 6),
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'ytick.labelsize': 'medium',
    'xtick.labelsize': 'medium',
    'legend.fontsize': 'large',
    'lines.linewidth': 1,
    'axes.prop_cycle': cycler('color',palette),
    'figure.facecolor': bg_color,
    'figure.edgecolor': bg_color,
    'axes.facecolor': bg_color,
    'text.color':grey_palette[1],
    'axes.labelcolor':grey_palette[1],
    'axes.edgecolor':grey_palette[1],
    'xtick.color':grey_palette[1],
    'ytick.color':grey_palette[1],
    'figure.dpi':150,
}

sns.set_theme(
    style='whitegrid',
    palette=sns.color_palette(palette),
    rc=custom_params
)


# # The Data
# 
# This week the dataset is a synthetic dataset based on [Pulsar Classification For Class Prediction
# ](https://www.kaggle.com/datasets/brsdincer/pulsar-classification-for-class-prediction). The competition is a **classification task** and the metric is the log-loss function.
# 
# ---
# Columns description from original  [Pulsar Classification For Class Prediction
# ](https://www.kaggle.com/datasets/brsdincer/pulsar-classification-for-class-prediction)
# 
# * Mean_Integrated: Mean of Observations
# * SD: Standard deviation of Observations
# * EK: Excess kurtosis of Observations
# * Skewness: In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. Skewness of Observations.
# * Mean _ DMSNR _ Curve: Mean of DM SNR CURVE of Observations
# * SD _ DMSNR _ Curve: Standard deviation of DM SNR CURVE of Observations
# * EK _ DMSNR _ Curve: Excess kurtosis of DM SNR CURVE of Observations
# * Skewness _ DMSNR _ Curve: Skewness of DM SNR CURVE of Observations
# * Class: Class 0 - 1
# 
# 
# ## What IS DM SNR CURVE:
# Radio waves emitted from pulsars reach earth after traveling long distances in space which is filled with free electrons. The important point is that pulsars emit a wide range of frequencies, and the amount by which the electrons slow down the wave depends on the frequency. Waves with higher frequency are sowed down less as compared to waves with higher frequency. It means dispersion.

# In[3]:


train_df = pd.read_csv('/kaggle/input/playground-series-s3e10/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s3e10/test.csv', index_col=0)

# load original dataset
original_df = pd.read_csv(
    '/kaggle/input/pulsar-classification-for-class-prediction/Pulsar.csv'
)

original_df.index.name = 'id'
original_df.index += test_df.index.max()+1

target = train_df.pop('Class')
original_target = original_df.pop('Class')
train_df.head()


# # The Label
# 
# As previous competitions, the class on the synthetic dataset follows the same proportion as the original dataset. Given that class is imbalaced, Stratified approch is paramount.
# 

# In[4]:


print('Synthetic')
display(target.value_counts(True))

print('\nOriginal')
display(original_target.value_counts(True))


# # EDA
# 
# # The data Size
# 
# 
# 
# **Insights**:
# 1. Dataset is fairly light but has lots of observations. We should try using neural networks and a fast linear model for quick iterations
# 2. Original dataset has less records that synthetic datase, beware of pseudo duplicates.

# In[5]:


print('Train shape:            ', train_df.shape)
print('Test shape:             ', test_df.shape)
print('Original Train shape:   ', original_df.shape)


# # The Duplicates
# 
# On previous competitions, participants have exploited duplicates to pos-process their predictions and achieve better score. On this competition, there are no duplicates.

# In[6]:


print('Duplicates on Synthetic Train: ', train_df.duplicated().sum())
print('Duplicates on Synthetic Test:  ', train_df.duplicated().sum())
print('Duplicates on Original:        ', train_df.duplicated().sum())


# # The Pseudo Duplicates
# 
# Given that we have only numeric features, its hard to find strict duplicates. What if we round our predictions to add some noise and verify if we have records with the exact information?
# 
# ## Rounding
# **Insights**:
# - By rounding to zero decimals we have 4240 duplicates. Contrary to past editions, this time the duplicates do not come in pairs. A postprocessing can be done in duplicates.
# - Ceil and floor functions both generate a different number of duplicates, any post-processing technique sould be verified over this three aproaches.

# In[7]:


dfs_label = ['Train', 'Oiginal', 'Test']
dfs = [train_df, original_df, test_df]

print(f'{"# Decimals":>10} {"Train":>10} {"Original":>10} {"Test":>10}')
for i in range(0, 3):
    dups = [i]
    for df in dfs:
        dups.append(df.round(i).duplicated().sum())
        
    print('{:10} {:10} {:10} {:10}'.format(*dups))


# ## Using np.ceil

# In[8]:


dfs_label = ['Train', 'Oiginal', 'Test']
dfs = [train_df, original_df, test_df]

print(f'{"Decimals":>10} {"Train":>10} {"Original":>10} {"Test":>10}')
for i in range(0, 3):
    dups = [i]
    for df in dfs:
        dups.append((np.ceil(df*10**i)/10**i).duplicated().sum())
        
    print('{:10} {:10} {:10} {:10}'.format(*dups))


# ## Using np.floor

# In[9]:


dfs_label = ['Train', 'Oiginal', 'Test']
dfs = [train_df, original_df, test_df]

print(f'{"Decimals":>10} {"Train":>10} {"Original":>10} {"Test":>10}')
for i in range(0, 3):
    dups = [i]
    for df in dfs:
        dups.append((np.floor(df*10**i)/10**i).duplicated().sum())
        
    print('{:10} {:10} {:10} {:10}'.format(*dups))


# # The TSNE
# 
# TSNE plots shows that classes are easily separated which means that we should expect a really high accuracy

# In[10]:


from openTSNE import TSNE

tsne = TSNE()
train_tsne = tsne.fit(train_df.to_numpy())
fig, ax = plt.subplots(figsize=(16, 16))
sns.scatterplot(x=train_tsne[:, 0], y=train_tsne[:, 1], hue=target, ax=ax)
ax.legend(fontsize=6, title='Class')
ax.set_title('2 Components - TSNE');


# # The Distributions
# 
# IIt's important to look at how data is spread out, as it can tell us if there are any unusual values in the data and if we need to adjust it. In the next part, we'll use simple graphs to see how the different parts of the data are related to each other, including the main thing we're interested in and any differences between the real and fake data.
# 
# 
# **Insights**:
# 
# 1. Not all features from synthetic have the same shape as original dataset. We should do a KS test to quantify which are the most different.
# 2. Some feature exhibits clear patterns with respect to the target for example a mean integrated that is less that 80 has a clear tendency to a positive Class. Tree-based models will easily spot these relationships, Linear models might require indicator columns to fully exploit the pattern.
# 3. `SD_DMSNR_CURVE` and `EK_DMSNR_Curve` has a non-linear relationship with respect to the target, I would start looking at this variables to check for any feature-crossing.
# 4. Clip EK, as low values won't provide any information.
# 5. Outliers distributions are not the same for train and original.
# 

# In[11]:


def plot_continous(feature, ax):
    temp = total_df.copy()
#     temp[feature] = temp[feature].clip(upper=temp[feature].quantile(0.99))
    
    sns.histplot(data=temp, x=feature,
                hue='set',ax=ax, hue_order=hue_labels,
                common_norm=False, **histplot_hyperparams)
    
    ax_2 = ax.twinx()
    ax_2 = plot_dot_continous(
        total_df.query('set=="train"'),
        feature, target, ax_2,
        color='#78BC61', df_set='train'
    )
    
    ax_2 = plot_dot_continous(
        total_df.query('set=="original"'),
        feature, original_target, ax_2,
        color='#FF7F50', df_set='original'
    )
    
    return ax_2
        
    
def plot_dot_continous(
    df, column, target, ax_2,
    show_yticks=False, color='green',
    df_set='train'
):

    bins = pd.cut(df[column], bins=n_bins)
    bins = pd.IntervalIndex(bins)
    bins = (bins.left + bins.right) / 2
    target = target.groupby(bins).mean()
    ax_2.plot(
        target.index,
        target, linestyle='',
        marker='.', color=color,
        label=f'Mean {df_set} {target.name}'
    )
    ax_2.grid(visible=False)
    
    if not show_yticks:
        ax_2.get_yaxis().set_ticks([])
    
    return ax_2


total_df = pd.concat([
    train_df.assign(set='train'),
    test_df.assign(set='test'),
    original_df.assign(set='original'),
], ignore_index=True)

total_df.reset_index(drop=True, inplace=True)
hue_labels = ['train', 'test', 'original']

numeric_features = [
    'Mean_Integrated', 'SD', 'EK', 'Skewness',
    'Mean_DMSNR_Curve', 'SD_DMSNR_Curve',
    'EK_DMSNR_Curve', 'Skewness_DMSNR_Curve'
]

n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}
line_style='--'

columns =  numeric_features
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*5))
ax = ax.flatten()

for i, column in enumerate(columns):
    ax2 = plot_continous(column, ax[i])
    # titles
    ax[i].set_title(f'{column} Distribution', pad=60);
    ax[i].set_xlabel(None)

    handles, labels = [], []
    plot_axes = [ax[i]]
    
#     Set legend for each plot
    for plot_ax in [ax[i], ax2]:
        if plot_ax.get_legend() is not None:
            handles += plot_ax.get_legend().legendHandles
            labels += [x.get_text() for x in plot_ax.get_legend().get_texts()]
        else:
            handles += plot_ax.get_legend_handles_labels()[0]
            labels += plot_ax.get_legend_handles_labels()[1]
            
    ax[i].legend(
        handles, labels,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.2), ncol=3,
        loc='upper center'
    )

for i in range(i+1, len(ax)):
    ax[i].axis('off') 

plt.tight_layout()


# # The Adversarial Validation
# 
# **Insights**
# 1. Most of the p-values from train vs Original (and test vs original), are zero, this is no surprise with the amount of data that we have. The KS-statistic is pretty low for much of the features. Maybe it is better to train a model to classify whether the feature is generated or not.
# 2. Train and test have low statistics and large p-values, features are the same on this two datasets.

# In[12]:


from scipy.stats import ks_2samp
cm = sns.light_palette(palette[1], as_cmap=True)

train_test_ks = {}
train_original_ks = {}
test_original_ks = {}

for feature in numeric_features:
    train_feature = total_df.query('set=="train"')[feature]
    test_feature = total_df.query('set=="test"')[feature]
    original_feature = total_df.query('set=="original"')[feature]
    
    train_test_ks[feature] = ks_2samp(train_feature, test_feature)
    train_original_ks[feature] = ks_2samp(train_feature, original_feature)
    test_original_ks[feature] = ks_2samp(test_feature, original_feature)
    

print('Train | Test')
display(pd.DataFrame.from_dict(
    train_test_ks, orient='index'
).style.background_gradient(cmap=cm))

print('\nTrain | Original')
display(pd.DataFrame.from_dict(
    train_original_ks, orient='index'
).style.background_gradient(cmap=cm))


print('\nTest | Original')
display(pd.DataFrame.from_dict(
    test_original_ks, orient='index'
).style.background_gradient(cmap=cm))


# # The Correlations
# 
# Usually I don't use correlations when doing a model, however over past playground series I have found that correlations are a good proxy for feature selection if a pair o features has a really strong correlation. This time it seems that is the case.
# 
# **Insights**
# 1. There are a lot of feature correlations. Use [DropCorrelatedFeatures
# ](https://feature-engine.trainindata.com/en/latest/user_guide/selection/DropCorrelatedFeatures.html) as a feature selection method.
# 2. If features have really high correlations a linear model may have troubles to converge.

# In[13]:


import warnings

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette[:3])
warnings.filterwarnings('ignore')
fig, ax = plt.subplots(1, 3, figsize=(20, 20))
float_types = [np.int64, np.float16, np.float32, np.float64]
float_columns = train_df.select_dtypes(include=float_types).columns
cbar_ax = fig.add_axes([.91, .39, .01, .2])

names = ['Train', 'Original']
for i, df in enumerate([train_df, original_df]):
    
    corr = df[float_columns].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, cmap=cmap,
        vmax=1, vmin=-1,
        center=0, annot=False, fmt='.3f',
        square=True, linewidths=.5,
        ax=ax[i],
        cbar=False,
        cbar_ax=None
    );

    ax[i].set_title(f'Correlation matrix for {names[i]} df', fontsize=14)

df = test_df
float_columns = test_df.select_dtypes(include=float_types).columns
corr = test_df[float_columns].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, cmap=cmap,
    vmax=1, vmin=-1,
    center=0, annot=False, fmt='.3f',
    square=True, linewidths=.5,
    cbar_kws={"shrink":.5, 'orientation':'vertical'},
    ax=ax[2],
    cbar=True,
    cbar_ax=cbar_ax
);
ax[2].set_title(f'Correlation matrix for Test', fontsize=14)
fig.tight_layout(rect=[0, 0, .9, 1]);


# # The Feature Engineering

# In[14]:


def fe(df):
    df['EK * EK_DMSNR_Curve'] = df.eval('EK * EK_DMSNR_Curve')
    df['EK * SD_DMSNR_Curve'] = df.eval('EK * SD_DMSNR_Curve')
    return df

train_df = fe(train_df)
test_df = fe(test_df)
original_df = fe(original_df)


# # The Baseline
# 
# In this code cell I'm implementing a LGBM model on which each fold includes the entire original dataset. This approach has suited me in previous playground competitions because original dataset is not exactly the same as sythentic and the optimization process should minimize the validation logloss that has minimal difference with test set.
# 
# **Insights**
# 1. The model is barely optimized but has little signals of overfitting.

# In[15]:


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

lgbm_params = {
    "max_depth":3,
    "reg_alpha":3.2975017728647247,
    "subsample":0.9641778045124232,
    "num_leaves":24,
    "reg_lambda":14.025493393231748,
    "n_estimators":1000,
    "subsample_freq":24,
    "min_child_samples":64
}

n_repeats=3
cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=n_repeats,
    random_state=2023
)

features = train_df.columns
X = train_df[features]
X_ts = test_df[features]
y = target

res = []
tr_log_losses = []
vl_log_losses = []
test_preds = []
models = []
oof_preds = pd.Series(0, index=train_df.index)
start = time()

for fold, (tr_ix, vl_ix) in enumerate(cv.split(train_df, target)):
    start_fold = time()
    X_tr, y_tr = X.loc[tr_ix].copy(), y.loc[tr_ix]
    X_vl, y_vl = X.loc[vl_ix].copy(), y.loc[vl_ix]
    X_ts = test_df[features].copy()

    # concat orginal df
    X_tr = pd.concat([X_tr, original_df[features]])
    y_tr = pd.concat([y_tr, original_target])

    scaler = SklearnTransformerWrapper(
        RobustScaler(),
        variables=list(features)
    )

    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_ts = scaler.transform(X_ts)

    X_tr['is_generated'] = (~X_tr.index.isin(original_df.index))
    X_vl['is_generated'] = 1
    X_ts['is_generated'] = 1

    model = LGBMClassifier(**lgbm_params, random_state=2023)
    model.fit(X_tr, y_tr,
              eval_set=(X_vl, y_vl),
              eval_metric='logloss',
              callbacks=[
                  log_evaluation(0), 
                  early_stopping(
                      50, verbose=False,
                      first_metric_only=True
                  )
              ]
             )

    y_pred_tr = model.predict_proba(X_tr)[:, 1]
    y_pred_vl = model.predict_proba(X_vl)[:, 1]
    oof_preds.iloc[vl_ix] += y_pred_vl
    test_preds.append(model.predict_proba(X_ts)[:, 1])

    tr_log_losses.append(log_loss(y_tr, y_pred_tr))
    vl_log_losses.append(log_loss(y_vl, y_pred_vl))
    models.append(model)

    print('_' * 30)
    print(f'Fold: {fold:<2} | Time     : {timedelta(seconds=int(time()-start))}')
    print(f'Fold Train logloss  : {Fore.BLUE}{tr_log_losses[-1]:.4f}{Style.RESET_ALL}')
    print(f'Fold Val logloss    : {Fore.BLUE}{vl_log_losses[-1]:.4f}{Style.RESET_ALL}')
    print(f'Training Time       : {timedelta(seconds=int(time()-start_fold))}')
    print()

oof_preds /= n_repeats
print(f'Mean Train logloss: {Fore.GREEN}{np.mean(tr_log_losses)}{Style.RESET_ALL}')
print(f'Mean Val logloss:   {Fore.GREEN}{np.mean(vl_log_losses)}{Style.RESET_ALL}')
print(f'OOF logloss:        {Fore.GREEN}{log_loss(target, oof_preds)}{Style.RESET_ALL}')


# # The Feature Importance
# 
# The feature importances shows a high standard deviation, this is a sign that fold-models over really different and increasing the number of folds might make the models similar. On Playground series episode 7 the 2nd place model had 30 folds but the 1st place model had just 3 folds.
# 
# On this competition, and with such an imabalaced dataset, I'll use something between 5 and 10.

# In[16]:


splits = []
gains = []
for model in models:
    splits.append(
        model.booster_.feature_importance(importance_type='split')
    )
    
    gains.append(
        model.booster_.feature_importance(importance_type='gain')
    )

fig, ax = plt.subplots(2, 1, figsize=(16, 10))
split_importances = pd.DataFrame(
    splits,
    columns=features.to_list() + ['is_generated']
)

gains_importances = pd.DataFrame(
    gains,
    columns=features.to_list() + ['is_generated']
)

sns.barplot(
    pd.melt(split_importances), y='variable', x='value',
    color=palette[1], orient='h', ax=ax[0]
)

sns.barplot(
    pd.melt(gains_importances), y='variable', x='value',
    color=palette[1], orient='h', ax=ax[1]
)

ax[0].set_title('Number of Splits')
ax[1].set_title('Total Gain');
plt.tight_layout()


# # The postprocessing
# 
# In this section we can validate wether there is a gain on post-processing your predictions to get a better score. The ideas to try are:
# 
# ## Set predictions that are close to 1 or close to 0, to 1 and 0 respectively.
# From the plot below we can see that there is no improvement on setting a lower bound as the best bound (the one that minimizes logloss) is 0, i.e no changes are made. For the upper bound there is tiny improvement before reaching to a threshold of 1 moving the logloss from 0.031631 -> 0.031622, this improvement can be due to randomness.

# In[17]:


lower_bound_scores = {}
for low_bound in np.linspace(0, 0.02, 100):
    oof_copy = oof_preds.copy()
    oof_copy[oof_copy.le(low_bound)] = 0
    lower_bound_scores[low_bound] = log_loss(target, oof_copy)
    
upper_bound_scores = {}
for upper_bound in np.linspace(0.99, 1, 100):
    oof_copy = oof_preds.copy()
    oof_copy[oof_copy.ge(upper_bound)] = 1
    upper_bound_scores[upper_bound] = log_loss(target, oof_copy)
    
fig, ax = plt.subplots(1, 2)
ax[0].plot(lower_bound_scores.keys(), lower_bound_scores.values())
ax[1].plot(upper_bound_scores.keys(), upper_bound_scores.values())

best_low_th = min(lower_bound_scores, key=lower_bound_scores.get)
best_up_th = min(upper_bound_scores, key=upper_bound_scores.get)

ax[0].axvline(best_low_th, linestyle='--')
ax[1].axvline(best_up_th, linestyle='--')

ax[0].text(s=f'{best_low_th:.3f}', x=best_low_th, y=ax[0].get_ylim()[1])
ax[1].text(s=f'{best_up_th:.3f}', x=best_up_th, y=ax[1].get_ylim()[1])

ax[0].axhline(log_loss(target, oof_preds), linestyle='--')
ax[1].axhline(log_loss(target, oof_preds), linestyle='--')

ax[0].set_ylabel('Threshold');
ax[1].set_ylabel('Threshold');

ax[0].set_ylabel('LogLoss');
ax[1].set_ylabel('LogLoss');

ax[0].set_title('Lower bound');
ax[1].set_title('Upper bound');


# ## Find the best threshold and update each side of the plot
# 
# There are several criterias for the "best threshold" for example, the two most commons are one that maximizes f1, maximizes accuracy or maximizes balanced accuracy. In this case, dataset is imabalaced so the second option is not viable.
# 
# **Insights**:
# - Best threshold according to f1 is not the same threshold according the balanced_accuracy
# - Both method can improve the score by 0.00002 which is again a really low value and the results can be due to randomness

# In[18]:


from sklearn.metrics import f1_score, balanced_accuracy_score

f1 = {}
balanced_accuracy = {}
for th in np.linspace(0, 1, 500):
    f1[th] = f1_score(target, oof_preds.ge(th))
    balanced_accuracy[th] = balanced_accuracy_score(target, oof_preds.ge(th))
    
fig, ax = plt.subplots()
ax.plot(f1.keys(), f1.values(), label='f1')
ax.plot(balanced_accuracy.keys(), balanced_accuracy.values(), label='balanced_accuracy')

best_f1_th = max(f1, key=f1.get)
best_ba_th = max(balanced_accuracy, key=balanced_accuracy.get)

ax.axvline(best_f1_th, linestyle='--', color=palette[0])
ax.axvline(best_ba_th, linestyle='--', color=palette[1])

ax.text(s=f'{best_f1_th:.3f}', x=best_f1_th, y=ax.get_ylim()[1])
ax.text(s=f'{best_ba_th:.3f}', x=best_ba_th, y=ax.get_ylim()[1])

ax.set_xlabel('Threshold')
ax.legend()

f1_scores = {}
for i in np.linspace(0, 0.001, 100):
    oof_copy = oof_preds.copy()
    oof_copy[oof_copy < best_f1_th] -= i
    oof_copy[oof_copy > best_f1_th] += i
    
    f1_scores[i] = log_loss(target, oof_copy)
    
f1_key = min(f1_scores, key=f1_scores.get)


ba_scores = {}
for i in np.linspace(0, 0.001, 100):
    oof_copy = oof_preds.copy()
    oof_copy[oof_copy < best_ba_th] -= i
    oof_copy[oof_copy > best_ba_th] += i
    
    ba_scores[i] = log_loss(target, oof_copy)
    
ba_key = min(ba_scores, key=ba_scores.get)
print(f'{"Method":>15} {"Base Score":>15} {"Best Score":>15} {"Diference":>15}')
print(f'{"f1":>15} {f1_scores[0]:15.6f} {f1_scores[f1_key]:15.6f} {f1_scores[0] - f1_scores[f1_key]:15.6f}')
print(f'{"b. accuracy":>15} {ba_scores[0]:15.6f} {ba_scores[ba_key]:15.6f} {ba_scores[0] - ba_scores[ba_key]:15.6f}')


# ## Ceil and Floor

# In[19]:


scores_lows = {}
for th in np.linspace(0, 0.5, 1000):
    oof_preds_ = oof_preds.copy()
    oof_preds_[oof_preds_ < th] = np.floor(oof_preds_[oof_preds_ < th] * 10**4)/10**4
    scores_lows[th] = log_loss(target, oof_preds_)
    
scores_high = {}
for th in np.linspace(0.5, 1, 1000):
    oof_preds_ = oof_preds.copy()
    oof_preds_[oof_preds_ > th] = np.ceil(oof_preds_[oof_preds_ > th] * 10**4)/10**4
    scores_high[th] = log_loss(target, oof_preds_)
    
print(
    'Ceil:     ',
    f'{log_loss(target, oof_preds):.6f}',
    f'{scores_lows[min(scores_lows, key=scores_lows.get)]:.6f}'
)

print(
    'Floor:     '
    f'{log_loss(target, oof_preds):.6f}',
    f'{scores_high[min(scores_high, key=scores_high.get)]:.6f}'
)

best_low_th = min(scores_lows, key=scores_lows.get)
best_high_th = min(scores_high, key=scores_high.get)

oof_preds_ = oof_preds.copy()
oof_preds_[oof_preds_ < best_low_th] = np.floor(
    oof_preds_[oof_preds_ < best_low_th] * 10**4
)/10**4

oof_preds_[oof_preds_ > best_high_th] = np.ceil(
    oof_preds_[oof_preds_ > best_high_th] * 10**4
)/10**4

print(
    'Combined: ',
    f'{log_loss(target, oof_preds):.6f}',
    f'{log_loss(target, oof_preds_):.6f}'
)


# # The Calibration
# 
# Calibration can potentially improve the score of log loss by post-processing the prediction in such way that the logloss is slighty minimized. In this example, isotonic calibration can improve the oof score by a tiny amount of ~ -0.0003

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

lr = LogisticRegression(C=99999999999, solver='liblinear', max_iter=1000)
lr.fit(oof_preds.values.reshape(-1, 1), target)
lr_preds_calibrated = lr.predict_proba(oof_preds.values.reshape(-1, 1))[:,1]

isor = IsotonicRegression(out_of_bounds='clip')
isor.fit(oof_preds.values.reshape(-1, 1), target)
isor_preds_calibrated = isor.predict(oof_preds.values.reshape(-1, 1))
isor_preds_calibrated = pd.Series(isor_preds_calibrated, index=target.index)

print('Plat Calibration Score    : ', log_loss(target, lr_preds_calibrated))
print('Isotonic Calibration Score: ', log_loss(target, isor_preds_calibrated))


# # The Submission
# 
# How to submit using the mean prediction of fold-models.

# In[21]:


submission = pd.read_csv(
    '/kaggle/input/playground-series-s3e10/sample_submission.csv',
    index_col=0
)

submission.Class = np.mean(test_preds, axis=0)
submission.Class = isor.predict(submission.Class)
submission.to_csv('submission.csv')


# # The Comparison OOF vs Test

# In[22]:


fig, ax = plt.subplots(1, 2)
lims = [(0, 0.1), (0.9, 1)]

for i, (x_i, x_o) in enumerate(lims):
    sns.histplot(
        isor_preds_calibrated[isor_preds_calibrated.between(x_i, x_o)],
        color=palette[0], ax=ax[i], label='oof',
        stat='percent',
        alpha=0.2, kde=True
    )

    sns.histplot(
        submission.Class[submission.Class.between(x_i, x_o)],
        color=palette[1], ax=ax[i], label='test',
        stat='percent',
        alpha=0.2, kde=True
    )

    ax[i].legend();

