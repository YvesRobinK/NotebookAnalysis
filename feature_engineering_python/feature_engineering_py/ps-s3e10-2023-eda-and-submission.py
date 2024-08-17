#!/usr/bin/env python
# coding: utf-8

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Libraries</p>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from pandas.io.formats.style import Styler
import math

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()

rc = {
    "axes.facecolor": "#FFFEF8",
    "figure.facecolor": "#FFFEF8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
grn = Style.BRIGHT + Fore.GREEN
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL

import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import log_loss

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1D8-X_32VIL89W80LHpUWyjRH2jasAYki"/>
# </p>

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Intro</p>

# This Kaggle notebook is aimed at providing a comprehensive exploratory data analysis (EDA) for the given dataset, with the ultimate goal of making informed decisions and recommendations before diving into modeling. 
# >Through this EDA, we will gain a deeper understanding of the data structure, missing values, relationships between variables, and any patterns or anomalies that could impact our modeling process. By performing a thorough EDA, we can identify potential roadblocks and make necessary pre-processing decisions that will improve the performance and accuracy of our models. So, buckle up, and let's embark on this journey of discovering insights and valuable information from the data to drive better modeling decisions.

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Data</p>

# **The dataset** for this competition (both train and test) was generated from a deep learning model trained on the [Pulsar Classification For Class Prediction](https://www.kaggle.com/datasets/brsdincer/pulsar-classification-for-class-prediction). The goal is to predict `0` or `1` (is pulsar or not).
# 
# >[A pulsar (from pulsating radio source)](https://en.wikipedia.org/wiki/Pulsar) is a highly magnetized rotating neutron star that emits beams of electromagnetic radiation out of its magnetic poles.
# 
#  
# There are 9 independent variables (including `id`):
# 
# * `Mean_Integrated`: Mean of Observations.
# * `SD`: Standard deviation of Observations.
# * `EK`: Excess kurtosis of Observations.
# * `Skewness`: In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. Skewness of Observations.
# * `Mean_DMSNR_Curve`: Mean of DM SNR CURVE of Observations.
# * `SD_DMSNR_Curve`: Standard deviation of DM SNR CURVE of Observations.
# * `EK_DMSNR_Curve`: Excess kurtosis of DM SNR CURVE of Observations.
# * `Skewness_DMSNR_Curve`: Skewness of DM SNR CURVE of Observations
# 
# Target varibale:
# * `Class`:  `0` is not pulsar, `1` is pulsar.
# 
# **Metrics**:
# * [LogLoss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
# 
# **DM SNR CURVE**:
# >Radio waves emitted from pulsars reach earth after traveling long distances in space which is filled with free electrons. The important point is that pulsars emit a wide range of frequencies, and the amount by which the electrons slow down the wave depends on the frequency. Waves with higher frequency are sowed down less as compared to waves with higher frequency. It means dispersion.

# In[2]:


PATH_ORIGIN = '/kaggle/input/pulsar-classification-for-class-prediction/Pulsar.csv'
PATH_TRAIN = '/kaggle/input/playground-series-s3e10/train.csv'
PATH_TEST = '/kaggle/input/playground-series-s3e10/test.csv'
PATH_SUB = '/kaggle/input/playground-series-s3e10/sample_submission.csv'

origin = pd.read_csv(PATH_ORIGIN)
train = pd.read_csv(PATH_TRAIN).drop(columns='id')
test = pd.read_csv(PATH_TEST).drop(columns='id')


# In[3]:


print(f'{gld}[INFO] Shapes:'
      f'{gld}\n[+] origin ->  {red}{origin.shape}'
      f'{gld}\n[+] train  -> {red}{train.shape}'
      f'{gld}\n[+] test   ->  {red}{test.shape}\n')

print(f'{gld}[INFO] Any missing values:'
      f'{gld}\n[+] origin -> {red}{origin.isna().any().any()}'
      f'{gld}\n[+] train  -> {red}{train.isna().any().any()}'
      f'{gld}\n[+] test   -> {red}{test.isna().any().any()}')


# #### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Train - Test</p>

# This notebook provides a code snippet of how effectively display dataframes in a tidy format using Pandas Styler class.
# 
# We are going to leverage CSS styling language to manipulate many parameters including colors, fonts, borders, background, format and make our tables interactive.
# 
# 
# **Reference**:
# * [Pandas Table Visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html).

# At this point, you will require a certain level of understanding in web development.
# 
# Primarily, you will have to modify the CSS of the `td`, `tr`, and `th` tags.
# 
# **You can refer** to the following materials to learn **HTML/CSS**:
# * [w3schools HTML Tutorial](https://www.w3schools.com/html/default.asp)
# * [w3schools CSS Reference](https://www.w3schools.com/cssref/index.php)

# In[4]:


def magnify(is_test: bool = False):
       base_color = '#457ea5'
       if is_test:
           highlight_target_row = []
       else:
           highlight_target_row = [dict(selector='tr:last-child',
                                        props=[('background-color', f'{base_color}'+'20')])]
           
       return [dict(selector="th",
                    props=[("font-size", "11pt"),
                           ('background-color', f'{base_color}'),
                           ('color', 'white'),
                           ('font-weight', 'bold'),
                           ('border-bottom', '0.1px solid white'),
                           ('border-left', '0.1px solid white'),
                           ('text-align', 'right')]),

               dict(selector='th.blank.level0', 
                   props=[('font-weight', 'bold'),
                          ('border-left', '1.7px solid white'),
                          ('background-color', 'white')]),

               dict(selector="td",
                    props=[('padding', "0.5em 1em"),
                           ('text-align', 'right')]),

               dict(selector="th:hover",
                    props=[("font-size", "14pt")]),

               dict(selector="tr:hover td:hover",
                    props=[('max-width', '250px'),
                           ('font-size', '14pt'),
                           ('color', f'{base_color}'),
                           ('font-weight', 'bold'),
                           ('background-color', 'white'),
                           ('border', f'1px dashed {base_color}')]),
               
                dict(selector="caption",
                     props=[(('caption-side', 'bottom'))])] + highlight_target_row

def stylize_min_max_count(pivot_table):
   """Waps the min_max_count pivot_table into the Styler.

       Args:
           df: |min_train| max_train |min_test |max_test |top10_counts_train |top_10_counts_train|

       Returns:
           s: the dataframe wrapped into Styler.
   """
   s = pivot_table
   # A formatting dictionary for controlling each column precision (.000 <-). 
   di_frmt = {(i if i.startswith('m') else i):
             ('{:.3f}' if i.startswith('m') else '{:}') for i in s.columns}

   s = s.style.set_table_styles(magnify(True))\
       .format(di_frmt)\
       .set_caption(f"The train and test datasets min, max, top10 values side by side (hover to magnify).")
   return s
 
   
def stylize_describe(df: pd.DataFrame, dataset_name: str = 'train', is_test: bool = False) -> Styler:
   """Applies .descibe() method to the df and wraps it into the Styler.
   
       Args:
           df: any dataframe (train/test/origin)
           dataset_name: default 'train'
           is_test: the bool parameter passed into magnify() function
                    in order to control the highlighting of the last row.
                    
       Returns:
           s: the dataframe wrapped into Styler.
   """
   s = df.describe().T
   # A formatting dictionary for controlling each column precision (.000 <-). 
   di_frmt = {(i if i == 'count' else i):
             ('{:.0f}' if i == 'count' else '{:.3f}') for i in s.columns}
   
   s = s.style.set_table_styles(magnify(is_test))\
       .format(di_frmt)\
       .set_caption(f"The {dataset_name} dataset descriptive statistics (hover to magnify).")
   return s

def stylize_simple(df: pd.DataFrame, caption: str) -> Styler:
   """Waps the min_max_count pivot_table into the Styler.

       Args:
           df: any dataframe (train/test/origin)

       Returns:
           s: the dataframe wrapped into Styler.
   """
   s = df
   s = s.style.set_table_styles(magnify(True)).set_caption(f"{caption}")
   return s

display(stylize_simple(train.head(4), 'The train dataset 3 top rows (hover to magnify).'))
display(stylize_describe(train))
display(stylize_describe(test, 'test', is_test=True))


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Min Max and Counts</p>

# In[5]:


cm = sns.light_palette('#457ea5', as_cmap=True)

counts_tr = pd.Series({ft: [train[ft].value_counts().round(3).iloc[:5].to_dict()] for ft in train.columns}, name='top_5_counts_train')
counts_te = pd.Series({ft: [test[ft].value_counts().round(3).iloc[:5].to_dict()] for ft in test.columns}, name='top_5_counts_test')
nunique_tr = train.nunique().rename('nunique_train')
nunique_te = test.nunique().rename('nunique_test')
nunique_te['Class'] = 0

min_max = train.describe().T[['min', 'max']].add_suffix('_train').join(test.describe().T[['min', 'max']].add_suffix('_test'))
stats_pivot = pd.concat([min_max, nunique_tr, nunique_te, counts_tr, counts_te], axis=1)
stylize_min_max_count(stats_pivot).background_gradient(cm, subset=['min_test', 'min_train', 'max_train', 'max_test'])


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Distributions</p>

# In[6]:


# kudos to @jcaliz
features = test.columns
n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}

columns = features
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    sns.kdeplot(
        train[column], label='Train',
        ax=ax[i], color=palette[0]
    )
    
    sns.kdeplot(
        test[column], label='Test',
        ax=ax[i], color=palette[1]
    )
    
    sns.kdeplot(
        origin[column], label='Original',
        ax=ax[i], color=palette[-4]
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
    
fig.suptitle(f'Dataset Feature Distributions\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=3)
plt.tight_layout()


# **Observations:**
# 
# * The distribution of the original dataset does not follow the synthetic one closely especially for the two last features above.
# * Although the train and test distributions are pefectly identical.
# 
# A synthetic dataset is a type of dataset created by generating new data that mimics the original data using various techniques. However, it is possible that the synthetic dataset features may not closely follow the original dataset distribution (our case). This can occur due to a variety of factors, such as using a different sampling technique, applying different data transformations, or introducing new features that were not present in the original dataset. When the synthetic dataset features do not closely follow the original dataset distribution, it can affect the performance of machine learning models trained on the origin data, as the models may not accurately capture the underlying patterns and relationships in the original data. Therefore, it is important to carefully evaluate the quality of both datasets before using them.
# 
# Let's take a look at the train dataset features against the target and target itself:

# In[7]:


fig = plt.figure(figsize=(6*6, 20), dpi=100)
for idx, col in enumerate(features):
    ax = plt.subplot(3, 3, idx + 1)
    sns.kdeplot(
        data=train, hue='Class', fill=True,
        x=col, palette=palette[:2], legend=False
    )
            
    ax.set_ylabel(''); ax.spines['top'].set_visible(False), 
    ax.set_xlabel(''); ax.spines['right'].set_visible(False)
    ax.set_title(f'{col}', loc='right', 
                 weight='bold', fontsize=20)

fig.suptitle(f'Features vs Class\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=3)
plt.tight_layout()
plt.show()


# In[8]:


def plot_count(df: pd.core.frame.DataFrame, col_list: list, title_name: str='Train') -> None:
    """Draws the pie and count plots for categorical variables.
    
    Args:
        df: train or test dataframes
        col_list: a list of the selected categorical variables.
        title_name: 'Train' or 'Test' (default 'Train')
        
    Returns:
        subplots of size (len(col_list), 2)
    """
    f, ax = plt.subplots(len(col_list), 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0)
    
    s1 = df[col_list].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = [palette[0], palette[0], '#ff781f', '#ff9752', '#ff9752']
    inner_colors = [palette[1], palette[1], '#ffa66b']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90,frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={ 'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size':13, 
        'weight': 'bold', 
        'color':'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%',explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', 
                               fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = [0, 1]
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=palette[:2], orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='black', 
                     fontweight='bold', fontsize=12)

#     plt.title(col_list)
    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col_list, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name} Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()    
#     plt.savefig('data/plot_count.png')
    plt.show()


# **Note**:
# * The distributions of features against `Class` `1` are either extremely flat or dense.
# 
# Let's take a look at the `Class` balance:

# In[9]:


plot_count(train, ['Class'], 'Train')
plot_count(origin, ['Class'], 'Original')


# **Note**:
# * Alright, we got a very imbalanced dataset. We are going to use [**StratifiedKFold**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) in order to preserve the percentage of samples for each class.

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Correlations</p>

# In[10]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    """Draws the correlation heatmap plot.
    
    Args:
        df: train or test dataframes
        title_name: 'Train' or 'Test' (default 'Train correlation')
        
    Returns:
        subplots of size (len(col_list), 2)
    """

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(10, 5))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap=palette[5:][::-2] + palette[1:2], annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(origin, 'Original Dataset Correlation')
plot_correlation_heatmap(train, 'Train Dataset Correlation')
plot_correlation_heatmap(train, 'Test Dataset Correlation')


# **Notes:**
# 
# * There are many highly correlated features (> |0.8|). We might end up dropping some of them or create simple interations.
# 

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Base XGB Model</p>

# In[11]:


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
    cm_display = ConfusionMatrixDisplay(cm).plot(cmap='Blues_r', ax=ax[0], values_format='5g')
    ax[0].grid(False)
    RocCurveDisplay.from_predictions(df.target, df.preds, color=palette[1], ax=ax[1])
    plt.tight_layout();
    
def get_mean_score(oof: np.array):
    """oof: ['val_idx', 'preds', 'target']"""
    oof = pd.DataFrame(np.concatenate(oof), columns=['id', 'preds', 'target']).set_index('id')
    oof.index = oof.index.astype(int)
    mean_val_score = log_loss(oof.target, oof.preds)
    return mean_val_score


# In[12]:


get_ipython().run_cell_magic('time', '', 'config = {\'SEED\': 42,\n          \'FOLDS\': 15,\n          \'N_ESTIMATORS\': 700}\n\nparams = {\'max_depth\': 4,\n          \'learning_rate\': 0.06,\n          \'colsample_bytree\': 0.67,\n          \'n_jobs\': -1,\n          \'objective\': \'binary:logistic\',\n          \'early_stopping_rounds\': 150,\n          \'verbosity\': 0,\n          \'eval_metric\': \'logloss\'}\n\n\nX, y = train.drop(columns=[\'Class\']), train.Class\ncv = model_selection.StratifiedKFold(n_splits=config[\'FOLDS\'],\n                                     shuffle=True,\n                                     random_state=config[\'SEED\'])\n\nfeature_importances_ = pd.DataFrame(index=test.columns)\neval_results_ = {}\nmodels_ = []\noof = []\n\nfor fold, (fit_idx, val_idx) in enumerate(cv.split(X, y)):\n    if (fold + 1) % 5 == 0 or (fold + 1) == 1:\n        print(f\'{"#" * 24} Training FOLD {fold + 1} {"#" * 24}\')\n\n    # Split the dataset according to the fold indexes.\n    X_fit = X.iloc[fit_idx]\n    X_val = X.iloc[val_idx]\n    y_fit = y.iloc[fit_idx]\n    y_val = y.iloc[val_idx]\n\n    # XGB .train() requires xgboost.DMatrix.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix\n    fit_set = xgb.DMatrix(X_fit, y_fit)\n    val_set = xgb.DMatrix(X_val, y_val)\n    watchlist = [(fit_set, \'fit\'), (val_set, \'val\')]\n\n    # Training.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training\n    eval_results_[fold] = {}\n    model = xgb.train(num_boost_round=config[\'N_ESTIMATORS\'],\n                      params=params,\n                      dtrain=fit_set,\n                      evals=watchlist,\n                      evals_result=eval_results_[fold],\n                      verbose_eval=False,\n                      callbacks=[EarlyStopping(params[\'early_stopping_rounds\'],\n                                               data_name=\'val\', save_best=True)])\n\n    val_preds = model.predict(val_set)\n    val_score = log_loss(y_val, val_preds)\n    best_iter = model.best_iteration\n\n    idx_pred_target = np.vstack([val_idx, val_preds, y_val]).T  # shape(len(val_idx), 3)\n    print(f\'{" " * 15} LogLoss:{blu}{val_score:.5f}{res} {" " * 6} best iteration  :{blu}{best_iter}{res}\')\n    \n    # Stores out-of-fold preds.\n    oof.append(idx_pred_target)\n\n    # Stores the feature importances\n    feature_importances_[f\'gain_{fold}\'] = model.get_score(importance_type=\'gain\').values()\n    feature_importances_[f\'split_{fold}\'] = model.get_score(importance_type=\'weight\').values()\n\n    # Stores the model\n    models_.append(model)\n\nmean_val_rmse = get_mean_score(oof)\nprint(f\'{"*" * 45}\\n{red}Mean{res} LogLoss: {red}{mean_val_rmse:.5f}\')\n')


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Training Summary</p>

# In[13]:


logloss_folds = pd.DataFrame.from_dict(eval_results_).T
fit_logloss = logloss_folds.fit.apply(lambda x: x['logloss'])
val_logloss = logloss_folds.val.apply(lambda x: x['logloss'])

fig, axes = plt.subplots(math.ceil(config['FOLDS']/3), 3, figsize=(30, 30), dpi=150)
ax = axes.flatten()
for i, (f, v, m) in enumerate(zip(fit_logloss, val_logloss, models_)): 
    sns.lineplot(f, color='#B90000', ax=ax[i], label='fit')
    sns.lineplot(v, color='#048BA8', ax=ax[i], label='val')
    ax[i].legend()
    ax[i].spines['top'].set_visible(False);
    ax[i].spines['right'].set_visible(False)
    ax[i].set_title(f'Fold {i}', fontdict={'fontweight': 'bold'})
    
    color =  ['#048BA8', palette[-3]]
    best_iter = m.best_iteration
    span_range = [[0, best_iter], [best_iter + 10, best_iter + params['early_stopping_rounds']]]
    
    for idx, sub_title in enumerate([f'Best\nIteration: {best_iter}', f'Early\n Stopping: {params["early_stopping_rounds"]}']):
        ax[i].annotate(sub_title,
                    xy=(sum(span_range[idx])/2 , 0.5),
                    xytext=(0,0), textcoords='offset points',
                    va="center", ha="center",
                    color="w", fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
        ax[i].axvspan(span_range[idx][0]-0.4,span_range[idx][1]+0.4,  color=color[idx], alpha=0.07)
    
    ax[i].set_xlim(0, best_iter + 20 + params["early_stopping_rounds"])
    ax[i].legend(bbox_to_anchor=(0.95, 1), loc='upper right', title='LogLoss')

plt.tight_layout();


# **Note**:
# * By increasing the number of iterations and reducing the learning rate, the model stopped finishing the training too early.
# * It might be a good idea to increase the number of iterations or decrease `early_stopping_rounds`.

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Feature importances and OOF errors</p>

# **There are** [several types of importance](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score) in the Xgboost. It can be computed in several different ways. The default type is gain if you construct model with scikit-learn like API (docs). When you access Booster object and get the importance with get_score method, then default is weight. You can check the type of the importance with xgb.importance_type.
# * The `gain` shows the average gain across all splits the feature is used in.
# * The `weight` shows  the number of times a feature is used to split the data across all trees.

# In[14]:


fi = feature_importances_
fi_gain = fi[[col for col in fi.columns if col.startswith('gain')]].mean(axis=1)
fi_splt = fi[[col for col in fi.columns if col.startswith('split')]].mean(axis=1)

fig, ax = plt.subplots(2, 1, figsize=(20, 10), dpi=150)
ax = ax.flatten()
# Split fi.
data_splt = fi_splt.sort_values(ascending=False)
sns.barplot(x=data_splt.values, y=data_splt.index, 
            color=palette[1], linestyle="-", width=0.5, errorbar='sd',
            linewidth=0.5, edgecolor="black", ax=ax[0])
ax[0].set_title(f'Feature Importance "Split"', fontdict={'fontweight': 'bold'})
ax[0].set(xlabel=None)

for s in ['right', 'top']:
    ax[0].spines[s].set_visible(False)
ax[0]
# Gain fi.    
data_gain = fi_splt.sort_values(ascending=False)
sns.barplot(x=data_gain.values, y=data_gain.index,
            color=palette[-3], linestyle="-", width=0.5, errorbar='sd',
            linewidth=0.5, edgecolor="black", ax=ax[1])
ax[1].set_title(f'Feature Importance "Gain"', fontdict={'fontweight': 'bold'})
ax[1].set(xlabel=None)

for s in ['right', 'top']:
    ax[1].spines[s].set_visible(False)


# In[15]:


show_confusion_roc(oof)


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Platt and Isotonic Probability Calibration</p>

# **Few notes of why we might need to use probability calibration techniques:**
# * A well-calibrated classifier can help to prevent incorrect decisions based on the classifier's predictions, especially if those decisions are based on threshold values for the predicted probabilities.
# * Poorly calibrated classifiers can lead to incorrect decisions, which can be especially problematic in high-stakes situations.
# * Probability calibration techniques can adjust the predicted probabilities to better reflect the true probabilities of positive outcomes, leading to more accurate and reliable predictions.
# 
# **The process:**
# 
# To assess the calibration of a binary classifier, we can plot a calibration curve that shows the predicted probabilities on the x-axis and the proportion of true positive outcomes for each predicted probability bin on the y-axis. The calibration curve can also show the proportion of true negative outcomes for each predicted probability bin. A perfectly calibrated classifier would have a calibration curve that is close to the diagonal line, indicating that the predicted probabilities accurately reflect the true probabilities of both positive and negative outcomes.
# 
# If the probabilities are not calibrated we can build in a sense the second level model fitting our predictions into Logistic regression or any other chosen algorithm against the target.
# After that we repeat the process by plotting the calibration curve and returning the new score.
# 
# **Methods:**
# 
# To calibrate a binary classifier, the data set of scores and their corresponding binary outcomes is used. The aim is to find a function that can accurately estimate the relationship between the scores and the true probabilities as determined empirically in the calibration set. There are several methods of calibration that can be used, including:
# 
# * Platt Scaling
# * Isotonic Regression
# * Beta Calibration
# * SplineCalib
# 
# The choice of calibration method depends on factors such as the size of the calibration set, the complexity of the classifier, and the desired level of calibration accuracy.
# 
# **Let's plot the calibration plot and histogram first:**
# 

# In[16]:


oof_df = pd.DataFrame(np.concatenate(oof), columns=['id', 'preds', 'target'])
oof_df.id = oof_df.id.astype(int)

def probability_calibration_plot(y_true=oof_df.target,
                                 y_preds=oof_df.preds,
                                 y_cali=None,
                                 n_bins=30,
                                 yerr_c=1,
                                 xylim=1,
                                 tick=0.1,
                                 calib_method=''): 
    
    prob_true, prob_pred = calibration_curve(y_true, y_preds, n_bins=n_bins)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=120)
    ax = ax.flatten()
    ax[0].errorbar(x=prob_pred, y=prob_true, yerr=abs(prob_true - prob_pred) * yerr_c, fmt=".k", label='Actual',
                   color=palette[1], capthick=0.5, capsize=3, elinewidth=0.7, ecolor=palette[1])

    sns.lineplot(x=np.linspace(0, xylim, 11), y=np.linspace(0, xylim, 11), color=palette[-3],
                 label='Perfectly calibrated', ax=ax[0], linestyle='dashed')
    
    if isinstance(y_cali, np.ndarray):
        prob_true_, prob_pred_ = calibration_curve(y_true, y_cali, n_bins=n_bins)
        sns.lineplot(x=prob_pred_, y=prob_true_, color=palette[-5],
                     label=f'{calib_method} Calibration', ax=ax[0], linestyle='solid')
    
    sns.histplot(y_preds, bins=n_bins*5, color=palette[1], ax=ax[1])
    for i, _ in enumerate(ax):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].xaxis.grid(False)
        ax[i].yaxis.grid(True)

    ax[0].set_title(f'Probability calibration plot', fontdict={'fontweight': 'bold'})
    ax[1].set_title(f'Histogram of predictions', fontdict={'fontweight': 'bold'})

    ax[0].set_xticks(list(np.arange(0, xylim + tick, tick)))
    ax[0].set_yticks(list(np.arange(0, xylim + tick, tick)))
    ax[0].set(xlabel='predicted', ylabel='actual')
    fig.suptitle(f'Predictions in range {(0, xylim)}', ha='center',  fontweight='bold', fontsize=16)
    plt.tight_layout();
    

probability_calibration_plot()


# **Note**:
# 
# It might look like the classifier is not well calibrated in a range ~(0.35 < x < 0.85) and we need to focus on that. But the important thing is to look at the histogram which shows that routhly 90% of our predictions < 0.5 and we need to **zoom in**:

# In[17]:


probability_calibration_plot(y_true=oof_df.query('preds < 0.05').target,
                             y_preds=oof_df.query('preds < 0.05').preds,
                             n_bins=300,
                             yerr_c=0.3,
                             xylim=0.05,
                             tick=0.01)


# Now we can see that our predictions < 0.05 are not well calibrated either.
# So let's try to fix it:
# 
# * We are going to apply Platt Scaling and Isotonic Regression:
# 
# >The **Platt Scaling** method assumes that there is a logistic relationship between the scores generated by a binary classifier and the true probability of positive outcomes. This means that it fits the two parameters of a logistic regression, just like in logistic regression models.
# >
# >This method has a very restrictive set of possible functions and requires very little data. It originated from the observation that a logistic regression is a suitable calibration method for Support Vector Machines, based on theoretical arguments.
# >
# >Platt, J. (1999) showed that using a logistic regression model to predict the probability of positive outcomes can improve the calibration of Support Vector Machines. The method involves training a logistic regression model on the scores generated by the binary classifier and their corresponding binary outcomes, and then using the logistic regression model to transform the predicted scores into calibrated probabilities.
# 
# >The **Isotonic Regression** method fits a piecewise constant, monotonically increasing function to map the scores generated by a binary classifier to probabilities. This method uses the Pool Adjacent Violators (PAV) algorithm and does not assume a particular parametric form.
# >
# >**Isotonic Regression** tends to perform better than Platt Scaling when there is sufficient data. However, it has a tendency **to overfit the calibration curve.**
# >
# > Zadrozny, B., & Elkan, C. (2001) proposed the use of Isotonic Regression for calibrating probability estimates from decision trees and naive Bayesian classifiers.

# In[18]:


lr = LogisticRegression(C=99999999999, solver='liblinear', max_iter=1000)
lr.fit(oof_df.preds.values.reshape(-1, 1), oof_df.target)
lr_preds_calibrated = lr.predict_proba(oof_df.preds.values.reshape(-1, 1))[:,1]
probability_calibration_plot(y_cali=lr_preds_calibrated, calib_method='Platt')

isor = IsotonicRegression(out_of_bounds='clip')
isor.fit(oof_df.preds.values.reshape(-1, 1), oof_df.target)
isor_preds_calibrated = isor.predict(oof_df.preds.values.reshape(-1, 1))

probability_calibration_plot(y_cali=isor_preds_calibrated, calib_method='Isotonic')
print(f'{gld}No calibration LogLoss:       {res}{log_loss(oof_df.target, oof_df.preds):.5f}')
print(f'{gld}Platt calibration LogLoss:    {red}{log_loss(oof_df.target, lr_preds_calibrated):.5f}{res}')
print(f'{gld}Isotonic calibration LogLoss: {grn}{log_loss(oof_df.target, isor_preds_calibrated):.5f}{res}\n\n')


# **Note**:
# 
# The **Isotonic Regression** has been found to yield better calibration performance than the **Platt Scaling**. The results showed that Isotonic Regression resulted in a **lower logloss** score than Logistic Regression after calibration, indicating more accurate probability estimates. However, the **Isotonic Regression tends to overfit** the calibration curve, which can result in unrealistic jumps. The Cross-validation techniques can be used to mitigate overfitting and ensure that the calibrated probabilities were reliable. Overall, the experiment demonstrated the potential benefits of using Isotonic Regression for calibrating binary classifiers.
# 
# **Homework:**
# * Explore **Beta Calibration** yourself
# * Explore **Spline Calibration** yourself
# 

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Submission</p>

# In[19]:


def predict(X):
    y = np.zeros(len(X))
    for model in tqdm(models_):
        y += model.predict(xgb.DMatrix(X))
    return y / len(models_)

predictions = predict(test)
sub = pd.read_csv(PATH_SUB)
sub.Class = predictions
sub.to_csv('submission.csv', index=False)
sub.head(3)


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Best Submission</p>
# **Don't forget to upvote.**

# In[20]:


best_sub = pd.read_csv('/kaggle/input/s3e10-submissions/XGB_LGBM_2.csv')
best_sub.to_csv('XGB_LGBM_2.csv', index=False)
best_sub.head(3)


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Acknowledgement</p>

# @jcaliz for .css and plotting ideas.

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#31709C; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #31709C">Outro and future work</p>
# 
# The feature engineering and adding more insightful context parts are on the way. 
# I hope to continue working on the dataset and probably shoot for a TOP.  Good luck in the competition!
